"""Pure Python EFMI physics serializer and rolling-palette INI generator.

This module deliberately imports neither bpy nor the add-on package. Stable
numeric vertex-group IDs are mapped to a compact, explicit draw palette; holes
left by deleted bones do not change surviving IDs. GPU limits are checked before
any artifact is returned. This is a model-space solver: no world inertia or
collision implementation is implied by the exported profile.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

MAX_BONES = 64
MAX_INSTANCES = 32
MAX_AUTHORING_ID = (1 << 31) - 1
SEGMENT_ROWS = 12
STATE_ROWS = 24
CHAIN_STATE_ROWS = 4
PROFILE_VERSION = 2


@dataclass(frozen=True)
class RuntimeArtifacts:
    ini_text: str
    files: dict[str, str]
    metadata: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    _require(number == value and minimum <= number <= maximum,
             f"{label} must be in {minimum}..{maximum}")
    return number


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(math.isfinite(result) and minimum <= result <= maximum,
             f"{label} must be finite and in {minimum}..{maximum}")
    return result


def _symbol(value: Any) -> str:
    _require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\\]*", value)),
             f"Invalid INI resource/command name: {value!r}")
    return value


def _matrix(value: Any, label: str) -> list[list[float]]:
    _require(isinstance(value, (tuple, list)) and len(value) == 4,
             f"{label} must be a row-major 4x4 affine matrix")
    result = []
    for row in value:
        _require(isinstance(row, (tuple, list)) and len(row) == 4,
                 f"{label} must be a row-major 4x4 affine matrix")
        result.append([_number(v, label, -1e7, 1e7) for v in row])
    _require(all(abs(a-b) < 1e-6 for a, b in zip(result[3], [0, 0, 0, 1])),
             f"{label} must be affine")
    return result


def _inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    inverse = [[matrix[c][r] for c in range(3)] + [0.] for r in range(3)]
    for r in range(3):
        inverse[r][3] = -sum(inverse[r][c] * matrix[c][3] for c in range(3))
    return inverse + [[0., 0., 0., 1.]]


def _rigid(matrix: list[list[float]], label: str) -> None:
    for a in range(3):
        for b in range(3):
            dot = sum(matrix[r][a] * matrix[r][b] for r in range(3))
            _require(abs(dot - (1. if a == b else 0.)) < 2e-4,
                     f"{label} must have unit orthogonal axes; apply scale before export")
    m = matrix
    det = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
           - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
           + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    _require(det > .999, f"{label} must be right-handed")


def _parameters(chain: Mapping[str, Any], segment: Mapping[str, Any]) -> dict[str, Any]:
    p = dict(chain.get("params", {}))
    p.update(segment.get("params", {}))
    gravity = p.get("gravity", [0., -9.8, 0.])
    _require(isinstance(gravity, (tuple, list)) and len(gravity) == 3,
             "gravity must have three model-space acceleration values")
    return {
        "gravity": [_number(v, "gravity", -1000., 1000.) for v in gravity],
        "damping": _number(p.get("damping", .08), "damping", 0., 1.),
        "stiffness": _number(p.get("stiffness", .18), "stiffness", 0., 1.),
        "cone_angle": _number(p.get("cone_angle", 45.), "cone_angle", 0., 180.),
        "wind_strength": _number(p.get("wind_strength", 0.), "wind_strength", 0., 10.),
        "translation_follow": _number(p.get("translation_follow", 1.), "translation_follow", 0., 1.),
        "max_velocity": _number(p.get("max_velocity", 30.), "max_velocity", .01, 1000.),
    }


def normalize_runtime_plan(plan: Mapping[str, Any], profile: Mapping[str, Any], *, allow_empty=False) -> dict[str, Any]:
    """Validate and return a JSON-only v2 profile; never modify input dictionaries.

    ``bones`` carry stable authoring global_id and an explicit dense palette_index.
    New authoring IDs start at 65536; legacy IDs in B..65535 remain supported.
    Only palette_index is an R16 GPU index. Each chain's segments are authoring
    IDs (or bone dictionaries), in root-to-tip order.
    Component offsets/counts refer only to the original native B palette.
    """
    adapter = dict(profile.get("adapter", {}))
    _require(adapter.get("kind", "efmi-rolling-buffer-v1") == "efmi-rolling-buffer-v1",
             "Only the verified EFMI rolling-buffer adapter is supported")
    contract = profile.get("input_contract", {})
    source = contract.get("validated_source", profile.get("bones_data_source"))
    _require(source in ("rolling", 1), "Unknown/CB posing source: capture a rolling-source profile first")
    _require(contract.get("matrix", "affine-3-float4-rows") == "affine-3-float4-rows",
             "Unsupported native matrix layout")
    _require(contract.get("bone_index_format", "R16G16B16A16_UINT") == "R16G16B16A16_UINT",
             "Unsupported blend-index format")
    B = _integer(plan.get("base_bone_count", adapter.get("original_bones", profile.get("bones_count"))),
                 "original bone count", 1, 65535)
    if "original_bones" in adapter:
        _require(adapter["original_bones"] == B, "Profile B differs from the exported skeleton")
    N = _integer(adapter.get("instances", profile.get("max_instance_count", 8)), "instance count", 1, MAX_INSTANCES)
    implicit = _integer(adapter.get("implicit_bones", 512), "implicit bone capacity", 0, 65535)
    components = []
    for c in profile.get("components", []):
        cid = _integer(c["id"], "component id", 0, 65535)
        offset = _integer(c.get("offset", c.get("bone_offset")), "component offset", 0, B)
        count = _integer(c.get("count", c.get("bone_count")), "component count", 0, B-offset)
        _require(c.get("declared_gpu_posed", True) is True,
                 f"Component {cid} is not a GPU rolling-palette component")
        components.append({"id": cid, "offset": offset, "count": count,
                           "draw_command": _symbol(c.get("draw_command", f"CommandList_Draw_Component{cid}"))})
    _require(bool(components), "At least one GPU component is required")
    _require(len({c["id"] for c in components}) == len(components), "Duplicate component IDs")
    _require(len({c["draw_command"] for c in components}) == len(components), "Duplicate component draw commands")
    for a in components:
        for b in components:
            _require(a is b or a["offset"] >= b["offset"]+b["count"] or b["offset"] >= a["offset"]+a["count"],
                     "Native component bone ranges overlap")
    raw_bones = list(plan.get("bones", []))
    raw_chains = list(plan.get("chains", []))
    minimum = 0 if allow_empty else 1
    _require(minimum <= len(raw_bones) <= MAX_BONES, f"A runtime supports {minimum}..{MAX_BONES} physics bones")
    _require(B + len(raw_bones) <= 65536, "Extended palette exceeds R16_UINT blend indices")
    _require((1 if raw_bones else 0) <= len(raw_chains) <= len(raw_bones), "Each chain needs at least one segment")
    by_id: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_bones):
        gid = _integer(raw.get("global_id"), "custom global ID", B, MAX_AUTHORING_ID)
        _require(gid not in by_id, "Duplicate custom global ID")
        bone = dict(raw)
        bone["global_id"] = gid
        bone["palette_index"] = _integer(raw.get("palette_index", B+index), "palette index", B, B+len(raw_bones)-1)
        by_id[gid] = bone
    _require(len({b["palette_index"] for b in by_id.values()}) == len(by_id), "Duplicate palette indices")
    used: set[int] = set()
    bones = []
    chains = []
    for ci, raw in enumerate(raw_chains):
        anchor_value = raw.get("anchor", raw.get("parent_global_id", raw.get("follow_group_id")))
        if isinstance(anchor_value, Mapping):
            anchor_value = anchor_value.get("global_id", anchor_value.get("index"))
        anchor = _integer(anchor_value, "chain native anchor", 0, B-1)
        candidates = [c for c in components if c["offset"] <= anchor < c["offset"]+c["count"]]
        _require(len(candidates) == 1, f"Anchor {anchor} does not belong to a hooked GPU component")
        component = candidates[0]
        if "component_id" in raw:
            _require(raw["component_id"] == component["id"], "Chain anchor/component mismatch")
        ids = []
        first = len(bones)
        previous = None
        for reference in raw.get("segments", []):
            gid = reference.get("global_id") if isinstance(reference, Mapping) else reference
            gid = _integer(gid, "segment global ID", B, MAX_AUTHORING_ID)
            _require(gid in by_id, f"Chain references missing bone {gid}")
            _require(gid not in used, f"Bone {gid} belongs to more than one chain")
            used.add(gid)
            bone = dict(by_id[gid])
            if isinstance(reference, Mapping):
                _require(reference.get("palette_index", bone["palette_index"]) == bone["palette_index"],
                         "Chain references cannot override the global-to-palette mapping")
                bone.update(reference)
            root_parent = -1 if raw.get("animation_anchor_group") else anchor
            expected_parent = root_parent if previous is None else previous["global_id"]
            parent = _integer(bone.get("parent_global_id", bone.get("parent", expected_parent)),
                              "segment parent ID", root_parent if root_parent == -1 else 0, MAX_AUTHORING_ID)
            _require(parent == expected_parent, f"Bone {gid} is not parented in root-to-tip chain order")
            rest = _matrix(bone.get("rest_matrix"), f"bone {gid} rest_matrix")
            _rigid(rest, f"bone {gid} rest_matrix")
            inv = _matrix(bone.get("inverse_bind", _inverse_rigid(rest)), f"bone {gid} inverse_bind")
            expected = _inverse_rigid(rest)
            _require(max(abs(inv[r][c]-expected[r][c]) for r in range(4) for c in range(4)) < 1e-3,
                     f"Bone {gid} inverse_bind disagrees with its edited rest frame")
            length = _number(bone.get("length"), f"bone {gid} length", 1.00001e-5, 1000.)
            if previous is not None:
                p = previous["rest_matrix"]
                tail = [p[r][3] + p[r][1]*previous["length"] for r in range(3)]
                error = math.sqrt(sum((rest[r][3]-tail[r])**2 for r in range(3)))
                _require(error < max(1e-3, length*1e-3), f"Bone {gid} head must connect to its parent tail")
            normalized = {"global_id": gid, "palette_index": bone["palette_index"],
                          "parent_global_id": expected_parent, "chain_index": ci,
                          "rest_matrix": rest, "inverse_bind": inv, "length": length,
                          "params": _parameters(raw, bone)}
            bones.append(normalized)
            ids.append(gid)
            previous = normalized
        _require(bool(ids), f"Chain {ci} is empty")
        chains.append({"anchor": anchor, "component_id": component["id"],
                       "component_offset": component["offset"], "component_count": component["count"],
                       "first_segment": first, "segment_count": len(ids), "segments": ids})
    _require(used == set(by_id), "Every custom bone must belong to exactly one chain")
    physics = profile.get("physics", {})
    settings = {
        "max_dt": _number(physics.get("max_dt", .1), "max dt", .001, .25),
        "teleport_distance": _number(physics.get("teleport_distance", 2.), "teleport distance", .001, 10000.),
        "max_gap_frames": _integer(physics.get("max_gap_frames", 8), "maximum frame gap", 1, 120),
        "substeps": _integer(physics.get("substeps", 4), "substeps", 1, 16),
        "max_abs_position": _number(physics.get("max_abs_position", 100000.), "position bound", 1., 1e7),
    }
    guard = profile.get("mod_guard", "$mod_enabled")
    route = profile.get("route_guard", "1")
    for value in (guard, route):
        _require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_$\\. ()!<>=&|+*/%\-]+", value)),
                 "INI guards must be single-line expressions without command separators")
    return {"version": PROFILE_VERSION, "base_bone_count": B, "added_bone_count": len(bones),
            "instance_count": N, "implicit_bone_capacity": implicit, "bones": bones, "chains": chains,
            "components": components, "physics": settings, "mod_guard": guard, "route_guard": route,
            "source_palette": _symbol(profile.get("source_palette", "ResourceMergedSkeletonDataRW")),
            "source_update_pool": _symbol(profile.get("source_update_pool", "Pool_MergedSkeleton_Instance_UpdateFrame")),
            "source_lod_pool": _symbol(profile.get("source_lod_pool", "Pool_MergedSkeleton_Instance_LodLevel")),
            "solver_space": "model", "one_frame_latency": True}


def profile_rows(normalized: Mapping[str, Any]) -> list[list[float]]:
    """SM5 ABI: four header float4s, two per chain, twelve per segment."""
    p = normalized
    B, K, N, I = (p[k] for k in ("base_bone_count", "added_bone_count", "instance_count", "implicit_bone_capacity"))
    chains, bones, physics = p["chains"], p["bones"], p["physics"]
    C = len(chains)
    rows = [[B, K, N, I], [C, 4, 4+2*C, SEGMENT_ROWS],
            [physics["max_dt"], physics["teleport_distance"], physics["max_gap_frames"], physics["substeps"]],
            [physics["max_abs_position"], PROFILE_VERSION, STATE_ROWS, CHAIN_STATE_ROWS]]
    for chain in chains:
        rows.extend([[chain["anchor"], chain["component_id"], chain["component_offset"], chain["component_count"]],
                     [chain["first_segment"], chain["segment_count"],
                      chain.get("animation_anchor_palette", -1) + 1, 0]])
    for bi, bone in enumerate(bones):
        params = bone["params"]
        # The existing GPU info.y is diagnostic only, never an index. Large
        # authoring IDs may round in float32; JSON and CPU ID mappings stay exact.
        # Only palette_index (info.x) is a GPU index; info.z stays a parent ordinal.
        rows.extend([[bone["palette_index"], bone["global_id"], bi-1 if bi != chains[bone["chain_index"]]["first_segment"] else -1, bone["length"]],
                     *bone["rest_matrix"][:3], *bone["inverse_bind"][:3],
                     [*params["gravity"], params["damping"]],
                     [params["stiffness"], math.radians(params["cone_angle"]), params["wind_strength"], params["translation_follow"]],
                     [params["max_velocity"], 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    _require(all(len(r) == 4 and all(math.isfinite(float(v)) for v in r) for r in rows), "Invalid GPU profile rows")
    return rows


def build_physics_runtime(plan: Mapping[str, Any], profile: Mapping[str, Any], *,
                          namespace: str = "EFMIPhysics", animation=None) -> RuntimeArtifacts:
    """Generate reviewable files; caller owns atomic filesystem publication.

    Append ``ini_text`` to the exported character INI and register each existing
    GPU component's callback using ``metadata['callback_wrappers']``. Wrapper
    command names are character-local; native draw commands are invoked once.
    """
    _require(isinstance(namespace, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", namespace)), "Invalid physics namespace")
    if animation:
        from ..efmi_animation.runtime import animation_constants, animation_files, animation_resources
    p = normalize_runtime_plan(plan, profile, allow_empty=bool(animation))
    B, K, N, I = (p[k] for k in ("base_bone_count", "added_bone_count", "instance_count", "implicit_bone_capacity"))
    C = len(p["chains"])
    added = K + (len(animation["bones"]) if animation else 0)
    _require(B + added <= 65536, "Extended palette exceeds R16_UINT blend indices")
    if animation:
        animation_bones = {bone["palette_index"]: bone for bone in animation["bones"]}
        _require(set(animation_bones) == set(range(B + K, B + added)) and 1 <= added-K <= 64,
                 "Invalid animation palette range")
        for source, chain in zip(plan.get("chains", []), p["chains"]):
            if "animation_anchor_palette" in source:
                palette = source["animation_anchor_palette"]
                _require(palette in animation_bones, "Physics chain references an unavailable animation bone")
                component = animation["chains"][animation_bones[palette]["chain_index"]]["component_id"]
                _require(any(item["id"] == component for item in p["components"]),
                         "Animation anchor component is missing from the rolling adapter")
                chain.update(animation_anchor_palette=palette, animation_anchor_component=component)
    linked = [index for index, chain in enumerate(p["chains"]) if "animation_anchor_palette" in chain]
    _require(len(linked) == sum(bool(chain.get("animation_anchor_group")) for chain in plan.get("chains", [])),
             "A physics chain with an animation anchor requires a matching exported animation bone")
    prefix = namespace
    resource = lambda name: f"Resource{prefix}_{name}"
    command = lambda name: f"CommandList{prefix}_{name}"
    shader = lambda name: f"CustomShader{prefix}_{name}"
    variable = lambda name: f"${prefix}_{name}"
    pool = lambda name: f"Pool{prefix}_{name}"
    v, r, cmd, sh, po = variable, resource, command, shader, pool
    rows = profile_rows(p)
    flat = " ".join(format(float(x), ".9g") for row in rows for x in row)
    groups = math.ceil((I*3 + 2*B*3)/64)
    updates, lods = p["source_update_pool"], p["source_lod_pool"]
    lines: list[str] = []
    def emit(text: str = "") -> None:
        lines.append(text.strip("\n"))

    def restore_constant_buffer(binding, saved, window, indent=""):
        offset, size = v(window + "_offset"), v(window + "_size")
        assignment = f"{binding} = ref {r('Save_' + saved)}"
        if animation:
            emit(f"{indent}if {offset} >= 0 && {size} > 0 && {size} <= 65536 && ({offset} & 255) == 0 && ({size} & 255) == 0")
            emit(f"{indent}    {assignment}->Region({offset}, {size})")
            emit(f"{indent}else")
            emit(f"{indent}    {assignment}")
            emit(f"{indent}endif")
        else:
            emit(indent + assignment)
    if animation:
        emit(f"; BEGIN {prefix} rolling-palette adapter\n; B={B}, physics={K}, animation={added-K}, instances={N}.\n[Constants]")
    else:
        emit(f"; BEGIN {prefix} model-space physics v2\n; Original B={B} stays unchanged. K={K}, chains={C}, N={N}; rolling source only.\n; Generated solver has one frame latency. World inertia/collisions are not implemented.\n[Constants]")
    for name, default in (("enabled", 1), ("simulate", 1), ("reset_epoch", 0), ("initialized", 0),
                          ("bound", 0), ("draw_ready", 0), ("slot", -1), ("index", 0),
                          ("generation", 0), ("anchor_ready", 0), ("allow_integrate", 0),
                          ("dt", 0), ("chain", 0), ("init_index", 0), ("saved_vs_cb_slot", -1),
                          ("unsupported_reason", 0)):
        emit(f"global {v(name)} = {default}")
    for ci in range(C):
        emit(f"global {v('chain_ready_'+str(ci))} = 0")
    if linked:
        emit(f"global {v('link_mode')} = 0")
    if animation:
        for target in ("cs_cb", "vs_cb"):
            emit(f"global {v(target + '_offset')} = -1\nglobal {v(target + '_size')} = -1")
        emit(animation_constants(p["mod_guard"], v("enabled") if C else None, lock_animation=bool(linked)))
    emit(f"\n[{cmd('Reset')}]\n{v('reset_epoch')} = ({v('reset_epoch')} + 1) & 1048575\n{v('initialized')} = 0\n{v('draw_ready')} = 0")
    state_defaults = (("LastSeenFrame", -1), ("LastPhysicsFrame", -1), ("LastPhysicsTime", 0),
                          ("LastPhysicsLod", -1), ("Generation", 0), ("InitFrame", -1),
                          ("InitGeneration", -1), ("ReadyFrame", -1), ("ReadyValue", 0))
    for name, default in state_defaults if C else ():
        emit(f"\n[{po(name)}]\npool_size = {N*C}\npool_variable_default_value = {default}")
    for name, count in (("ExtendedPalette", (I+2*(B+added)*N)*3), ("State", N*K*STATE_ROWS), ("ChainState", N*C*CHAIN_STATE_ROWS)):
        if not count:
            continue
        emit(f"\n[{r(name)}]\ntype = RWStructuredBuffer\nstride = 16\narray = {count}\nbind_flags = shader_resource unordered_access")
    emit(f"\n[{r('Profile')}]\ntype = StructuredBuffer\nstride = 16\narray = {len(rows)}\ndata = R32_FLOAT {flat}")
    emit(f"\n[{r('InstanceConfig')}]\ntype = RWBuffer\nformat = R32G32B32A32_UINT\narray = 16\nbind_flags = shader_resource unordered_access")
    saves = {"CS_CB0": "cs-cb0", "CS_T0": "cs-t0", "CS_T1": "cs-t1", "CS_U0": "cs-u0", "CS_U1": "cs-u1", "CS_U2": "cs-u2", "VS_T0": "vs-t0"}
    for name in (*saves, "VS_Config"):
        emit(f"\n[{r('Save_'+name)}]")
    if C:
        emit(f"\n[{cmd('Initialize')}]\nclear = {r('ExtendedPalette')}\nclear = {r('State')}\nclear = {r('ChainState')}\n{v('init_index')} = 0\nrun = {cmd('InitializeSlot')}\n{v('initialized')} = 1")
        emit(f"\n[{cmd('InitializeSlot')}]")
        if animation:
            # XXMI 1.3.3 limits nested commands to 256. Reset up to 16 slots
            # per call to support the complete hybrid palette capacity.
            for _ in range(min(N*C, 16)):
                emit(f"if {v('init_index')} < {N*C}")
                for name, default in state_defaults:
                    emit(f"    ${po(name)}[{v('init_index')}] = {default}")
                emit(f"    {v('init_index')} = {v('init_index')} + 1\nendif")
        else:
            for name, default in state_defaults:
                emit(f"${po(name)}[{v('init_index')}] = {default}")
            emit(f"{v('init_index')} = {v('init_index')} + 1")
        emit(f"if {v('init_index')} < {N*C}\n    run = {cmd('InitializeSlot')}\nendif")
    enabled = v("enabled")
    initialize = f"            if !{v('initialized')}\n                run = {cmd('Initialize')}\n            endif\n"
    update_clock = ""
    if animation:
        enabled = f"({enabled} || $EFMIAnimation_enabled)" if C and not linked else "$EFMIAnimation_enabled"
        initialize = initialize if C else ""
        update_clock = "\nrun = CommandListEFMIAnimationUpdateTime"
    emit(f"\n[{cmd('Begin')}]\n{v('bound')} = 0\n{v('draw_ready')} = 0\n{v('unsupported_reason')} = 0{update_clock}\nif {enabled} && ({p['mod_guard']}) && ({p['route_guard']}) && $\\EFMIv1\\merged_skeleton_connected\n    if $\\EFMIv1\\bones_data_source != 1 || $\\EFMIv1\\instance_config_vs_cb_slot < 1 || $\\EFMIv1\\instance_config_vs_cb_slot > 3\n        {v('unsupported_reason')} = 1\n    elif $\\EFMIv1\\bones_count != {B} || $\\EFMIv1\\instance_count != {N} || $\\EFMIv1\\cfg_ms_implicit_bones_count < 0 || $\\EFMIv1\\cfg_ms_implicit_bones_count > {I}\n        {v('unsupported_reason')} = 2\n    else\n        {v('slot')} = $\\EFMIv1\\merged_skeleton_instance_id\n        if {v('slot')} >= 0 && {v('slot')} < {N}\n{initialize}            run = {cmd('Bind')}\n            {v('bound')} = 1\n        else\n            {v('unsupported_reason')} = 3\n        endif\n    endif\nendif")
    # CPU frame/generation state is PER CHAIN. A fresh anchor in one component
    # must not authorize another chain whose component has not been posed.
    for ci, chain in enumerate(p["chains"]):
        component = chain.get("animation_anchor_component", chain["component_id"])
        component_offset = component * N
        link_setup = f"{v('link_mode')} = {int(ci in linked)}\n" if linked else ""
        gap, maxdt = p["physics"]["max_gap_frames"], p["physics"]["max_dt"]
        emit(f"\n[{cmd('PrepareChain'+str(ci))}]\n{v('chain')} = {ci}\n{v('index')} = {ci*N} + {v('slot')}\n{v('generation')} = ${po('Generation')}[{v('index')}]\n{link_setup}local $last_seen = ${po('LastSeenFrame')}[{v('index')}]\nif $last_seen >= 0 && (FRAME_NUMBER < $last_seen || FRAME_NUMBER - $last_seen > {gap})\n    {v('generation')} = ({v('generation')} + 1) & 1048575\n    ${po('InitFrame')}[{v('index')}] = -1\nendif\n${po('LastSeenFrame')}[{v('index')}] = FRAME_NUMBER\n{v('anchor_ready')} = ${updates}[{component_offset} + {v('slot')}] == FRAME_NUMBER\n{v('allow_integrate')} = {v('anchor_ready')} && $\\EFMIv1\\component_id == {component} && ${po('LastPhysicsFrame')}[{v('index')}] != FRAME_NUMBER\n{v('dt')} = 0\nif {v('allow_integrate')}\n    local $last_time = ${po('LastPhysicsTime')}[{v('index')}]\n    local $last_lod = ${po('LastPhysicsLod')}[{v('index')}]\n    local $current_lod = ${lods}[{component_offset} + {v('slot')}]\n    {v('dt')} = time - $last_time\n    if $last_lod != $current_lod || {v('dt')} < 0 || {v('dt')} > {maxdt}\n        {v('generation')} = ({v('generation')} + 1) & 1048575\n        ${po('InitFrame')}[{v('index')}] = -1\n        {v('dt')} = 0\n    endif\n    ${po('LastPhysicsFrame')}[{v('index')}] = FRAME_NUMBER\n    ${po('LastPhysicsTime')}[{v('index')}] = time\n    ${po('LastPhysicsLod')}[{v('index')}] = $current_lod\n    if ${po('InitFrame')}[{v('index')}] < 0 || ${po('InitGeneration')}[{v('index')}] != {v('generation')}\n        ${po('InitFrame')}[{v('index')}] = FRAME_NUMBER\n        ${po('InitGeneration')}[{v('index')}] = {v('generation')}\n    endif\nendif\n${po('Generation')}[{v('index')}] = {v('generation')}\n; Visibility is frozen by the full frame, even if a later anchor changes LoD.\nif ${po('ReadyFrame')}[{v('index')}] != FRAME_NUMBER\n    ${po('ReadyValue')}[{v('index')}] = 0\n    if ${po('InitFrame')}[{v('index')}] >= 0 && FRAME_NUMBER > ${po('InitFrame')}[{v('index')}] && ${po('InitGeneration')}[{v('index')}] == {v('generation')}\n        ${po('ReadyValue')}[{v('index')}] = 1\n    endif\n    ${po('ReadyFrame')}[{v('index')}] = FRAME_NUMBER\nendif\n{v('chain_ready_'+str(ci))} = ${po('ReadyValue')}[{v('index')}]\nrun = {sh('Prepare')}")
    emit(f"\n[{cmd('Bind')}]")
    for name, binding in saves.items():
        emit(f"{r('Save_'+name)} = ref {binding}")
    if animation:
        emit(f"{v('cs_cb_offset')} = cs-cb0->Offset\n{v('cs_cb_size')} = cs-cb0->Size")
    emit(f"{v('saved_vs_cb_slot')} = $\\EFMIv1\\instance_config_vs_cb_slot")
    for slot in (1, 2, 3):
        emit(f"{'if' if slot == 1 else 'elif'} {v('saved_vs_cb_slot')} == {slot}\n    {r('Save_VS_Config')} = ref vs-cb{slot}")
        if animation:
            emit(f"    {v('vs_cb_offset')} = vs-cb{slot}->Offset\n    {v('vs_cb_size')} = vs-cb{slot}->Size")
    emit("endif\n; Borrow exact views. Never retain a render target or backbuffer.\nvs-t0 = null\ncs-t0 = null\ncs-t1 = null\ncs-u0 = null\ncs-u1 = null\ncs-u2 = null")
    if animation:
        emit("run = CustomShaderEFMIAnimationPrepare")
        if C:
            emit(f"if {v('enabled')}")
    for ci in range(C):
        indent = "    " if animation else ""
        emit(f"{indent}run = {cmd('PrepareChain'+str(ci))}")
    if animation and C:
        emit("endif")
    ready_expression = "1" if animation else " && ".join(v('chain_ready_'+str(ci)) for ci in range(C))
    emit(f"{v('draw_ready')} = {ready_expression}\nrun = {sh('PatchConfig')}\ncs-u0 = null\ncs-u1 = null\ncs-u2 = null\ncs-t0 = null\ncs-t1 = null")
    restore_constant_buffer("cs-cb0", "CS_CB0", "cs_cb")
    for name in ("CS_T0", "CS_T1", "CS_U0", "CS_U1", "CS_U2"):
        emit(f"{saves[name]} = no_view_cache ref {r('Save_'+name)}")
    for slot in (1, 2, 3):
        emit(f"{'if' if slot == 1 else 'elif'} {v('saved_vs_cb_slot')} == {slot}\n    vs-cb{slot} = copy {r('InstanceConfig')}")
    emit(f"endif\nvs-t0 = ref {r('ExtendedPalette')}")
    # Only 0,1,2 IniParams are temporarily borrowed. No game CB region or source
    # view remains in our CS bindings after either compute dispatch.
    if animation:
        animation_groups = math.ceil((I * 3 + 2 * B * 3 + added) / 64)
        emit(animation_resources(p["source_palette"], r("ExtendedPalette"), v("slot"), animation_groups))
    for name in (("Prepare", "PatchConfig") if C else ("PatchConfig",)):
        emit(f"\n[{sh(name)}]\ncs = res/{prefix}/Shaders/{'prepare' if name == 'Prepare' else 'config_patch'}.hlsl\nrun = CommandList\\EFMIv1\\BackupIniParams0\nrun = CommandList\\EFMIv1\\BackupIniParams1\nlocal $bak_x2 = x2\nlocal $bak_y2 = y2\nlocal $bak_z2 = z2\nlocal $bak_w2 = w2\nx0 = FRAME_NUMBER & 1048575\ny0 = {v('slot')}\nz0 = {v('anchor_ready')}\nw0 = {v('allow_integrate')}\nx1 = {v('dt')}\ny1 = time\nz1 = {v('generation')}\nw1 = {v('reset_epoch')}\nx2 = {v('simulate')}\ny2 = $\\EFMIv1\\cfg_ms_implicit_bones_count\nz2 = {v('chain')}\nw2 = 0")
        if animation:
            # A negative width keeps the seeded native fallback for this
            # physics chain until its published frame becomes ready.
            extra_width = f"{added} * (2 * ${po('ReadyValue')}[{v('index')}] - 1)" if name == "Prepare" else str(added)
            emit(f"w2 = {extra_width}")
            if linked and name == "Prepare":
                emit(f"x2 = {v('simulate')} + 2 * {v('link_mode')}")
        if name == "Prepare":
            emit(f"cs-t0 = ref {p['source_palette']}\ncs-t1 = ref {r('Profile')}\ncs-u0 = ref {r('ExtendedPalette')}\ncs-u1 = ref {r('State')}\ncs-u2 = ref {r('ChainState')}\ndispatch = {1 if animation else groups}, 1, 1")
        else:
            emit(f"cs-cb0 = ref Resource\\EFMIv1\\Original_VS_CB->Region($\\EFMIv1\\instance_config_vs_cb_byte_offset + $\\EFMIv1\\draw_call_instance_id * $\\EFMIv1\\cfg_instance_config_byte_size, $\\EFMIv1\\cfg_instance_config_byte_size)\ncs-t1 = ref {r('Profile')}\ncs-u0 = ref {r('InstanceConfig')}\ndispatch = 1, 1, 1")
        emit("cs-u0 = null\ncs-u1 = null\ncs-u2 = null\ncs-t0 = null\ncs-t1 = null")
        if name == "PatchConfig" and not animation:
            emit(f"cs-cb0 = ref {r('Save_CS_CB0')}")
        emit("x2 = $bak_x2\ny2 = $bak_y2\nz2 = $bak_z2\nw2 = $bak_w2\nrun = CommandList\\EFMIv1\\RestoreIniParams0\nrun = CommandList\\EFMIv1\\RestoreIniParams1")
    emit(f"\n[{cmd('End')}]\nif {v('bound')}\n    vs-t0 = null")
    for slot in (1, 2, 3):
        emit(f"    {'if' if slot == 1 else 'elif'} {v('saved_vs_cb_slot')} == {slot}")
        restore_constant_buffer(f"vs-cb{slot}", "VS_Config", "vs_cb", "        ")
    emit(f"    endif\n    vs-t0 = no_view_cache ref {r('Save_VS_T0')}\n    ; Null alone keeps the backing resource alive in 3DMigoto. Replace every\n    ; borrowed reference with our safe nonempty IniParams resource, then null.")
    for name in (*saves, "VS_Config"):
        emit(f"    {r('Save_'+name)} = ref iniparams\n    {r('Save_'+name)} = null")
    emit(f"endif\n{v('bound')} = 0\n{v('draw_ready')} = 0")
    wrappers = {}
    for component in p["components"]:
        wrapper = cmd('Wrapper_Component'+str(component['id']))
        wrappers[component["draw_command"]] = wrapper
        emit(f"\n[{wrapper}]\nrun = {cmd('Begin')}\nrun = {component['draw_command']}\nrun = {cmd('End')}")
    emit(f"; END {prefix} {'rolling-palette adapter' if animation else 'model-space physics v2'}\n")
    ini = "\n".join(lines) + "\n"
    files = {f"res/{prefix}/Shaders/{name}.hlsl": (Path(__file__).parent / "shaders" / f"{name}.hlsl").read_text(encoding="utf-8")
             for name in (("prepare", "config_patch") if C else ("config_patch",))}
    if animation:
        files.update(animation_files(animation, p))
    files[f"res/{prefix}/profile.json"] = json.dumps(p, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    metadata = {"version": PROFILE_VERSION, "namespace": prefix, "callback_wrappers": wrappers,
                "chain_ready_variables": [v('chain_ready_'+str(ci)) for ci in range(C)],
                "draw_ready_variable": v('draw_ready'), "reset_command": cmd('Reset'),
                "bound_variable": v('bound'), "enabled_variable": v('enabled'), "simulate_variable": v('simulate'),
                "unsupported_reason_variable": v('unsupported_reason'),
                "global_to_palette": {b["global_id"]: b["palette_index"] for b in p["bones"]},
                "original_bones": B, "extended_bones": B+added, "instances": N,
                "profile_float4_rows": len(rows), "implicit_bone_capacity": I,
                "solver_space": "model", "one_frame_latency": True,
                "runtime_verification_pending": ["multi-chain game draw order", "resolution/fullscreen transition", "LoD switching"],
                "implemented": ["connected multi-segment chains", "native anchor follow", "length constraints", "damping", "gravity", "small wind", "spring return", "cone limit"],
                "unsupported": ["CB/unknown posing sources", "world inertia", "collision", "more than 64 extra bones", "more than 32 parallel instances"]}
    if animation:
        metadata["animation_enabled_variable"] = "$EFMIAnimation_enabled"
        metadata["animation_required_for_physics"] = bool(linked)
    return RuntimeArtifacts(ini, files, metadata)
