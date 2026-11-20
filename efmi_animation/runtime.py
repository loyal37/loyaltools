"""Float4 sample table and INI commands for the EFMI rolling-palette adapter."""

from pathlib import Path
import struct


def build_sample_buffer(animation, physics):
    """Pack data as little-endian float4s with exact row and bone ordinals."""
    bones = animation["bones"]
    chains = animation["chains"]
    animation_ordinals = {bone["palette_index"]: index + 1 for index, bone in enumerate(bones)}
    fallback = []
    for bone in physics["bones"]:
        chain = physics["chains"][bone["chain_index"]]
        link = animation_ordinals.get(chain.get("animation_anchor_palette"), 0)
        fallback.append((bone["palette_index"], chain["anchor"], link, 0))
    bone_start = 3 + len(chains)
    fallback_start = bone_start + 4 * len(bones)
    rows = [
        (physics["base_bone_count"], len(bones) + len(fallback),
         physics["instance_count"], physics["implicit_bone_capacity"]),
        (len(chains), len(bones), bone_start, fallback_start),
        (len(fallback), 0, 0, 0),
        *[(0, 0, 0, 0)] * len(chains),
    ]
    clip_bones = [[] for _ in chains]
    for bone in bones:
        clip = clip_bones[bone["chain_index"]]
        rows.append((bone["palette_index"], bone["chain_index"], bone["parent_index"], len(clip)))
        rows.extend(bone["inverse_bind"][:3])
        clip.append(bone)
    rows.extend(fallback)
    rows.extend((chain["anchor"], 0, 0, 0) for chain in chains)
    for chain in chains:
        rows.extend(chain["root_transform"][:3])
    for index, (chain, clip) in enumerate(zip(chains, clip_bones)):
        count = len(clip[0]["frames"])
        rate = chain["fps"] * chain["speed"] * chain["samples_per_frame"]
        period = 1
        if count > 1:
            period = count - 1
            if not chain["seamless_loop"]:
                period += chain["samples_per_frame"]
        rows[3 + index] = (len(rows), count, rate, period)
        for frame in zip(*(bone["frames"] for bone in clip)):
            for sample in frame:
                rows.extend([(*sample["translation"], 0), sample["rotation"], (*sample["scale"], 0)])
    packed = bytearray(len(rows) * 16)
    for index, row in enumerate(rows):
        struct.pack_into("<4f", packed, index * 16, *row)
    return bytes(packed)


def animation_files(animation, physics):
    return {
        "res/EFMIAnimation/samples.buf": build_sample_buffer(animation, physics),
        "res/EFMIAnimation/Shaders/prepare.hlsl": (
            Path(__file__).parent / "shaders" / "prepare.hlsl"
        ).read_text(encoding="utf-8"),
    }


def animation_constants(mod_guard, physics_enabled=None, *, lock_animation=False):
    toggle = "; Animation is required by an authored physics attachment." if lock_animation else f"""[KeyEFMIAnimationToggle]
condition = {mod_guard}
key = CTRL ALT F8
type = toggle
$EFMIAnimation_enabled = 1, 0"""
    return f"""global $EFMIAnimation_enabled = 1
global $EFMIAnimation_frame = -1
global $EFMIAnimation_start_time = -1
global $EFMIAnimation_time = 0
global $EFMIAnimation_previous_time = 0

{toggle}

{physics_toggle(mod_guard, physics_enabled) if physics_enabled else ''}

[Present]
post run = CommandListEFMIAnimationResetClock"""


def physics_toggle(mod_guard, enabled):
    return f"""[KeyEFMIAnimationPhysicsToggle]
condition = {mod_guard}
key = CTRL ALT F7
type = toggle
{enabled} = 1, 0"""


def animation_resources(source, expanded, slot_var, groups):
    return rf"""
[ResourceEFMIAnimationSamples]
type = StructuredBuffer
stride = 16
filename = res/EFMIAnimation/samples.buf

[CommandListEFMIAnimationResetClock]
if !$EFMIAnimation_enabled
    $EFMIAnimation_start_time = -1
    $EFMIAnimation_frame = -1
endif

[CommandListEFMIAnimationUpdateTime]
run = CommandListEFMIAnimationResetClock
if $EFMIAnimation_enabled && $EFMIAnimation_frame != FRAME_NUMBER
    local $now = time
    if $EFMIAnimation_start_time < 0 || $now - $EFMIAnimation_start_time < $EFMIAnimation_time
        $EFMIAnimation_start_time = $now
        $EFMIAnimation_time = 0
        $EFMIAnimation_previous_time = 0
    else
        $EFMIAnimation_previous_time = $EFMIAnimation_time
        $EFMIAnimation_time = $now - $EFMIAnimation_start_time
        if FRAME_NUMBER != $EFMIAnimation_frame + 1
            $EFMIAnimation_previous_time = $EFMIAnimation_time
        endif
    endif
    $EFMIAnimation_frame = FRAME_NUMBER
endif

[CustomShaderEFMIAnimationPrepare]
cs = res/EFMIAnimation/Shaders/prepare.hlsl
run = CommandList\EFMIv1\BackupIniParams0
run = CommandList\EFMIv1\BackupIniParams1
x0 = $EFMIAnimation_time
y0 = $EFMIAnimation_previous_time
z0 = {slot_var}
w0 = $EFMIAnimation_enabled
x1 = $\EFMIv1\cfg_ms_implicit_bones_count
cs-t0 = ref {source}
cs-t1 = ref ResourceEFMIAnimationSamples
cs-u0 = ref {expanded}
dispatch = {groups}, 1, 1
cs-u0 = null
cs-t1 = null
cs-t0 = null
run = CommandList\EFMIv1\RestoreIniParams0
run = CommandList\EFMIv1\RestoreIniParams1
"""
