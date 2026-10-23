# -*- coding: utf-8 -*-
"""Zenless Zone Zero (ZZMI) merged skeleton: workspace contract and runtime data.

This module has no ``bpy`` dependency.  Extraction, Blender import and the ZZMI
exporter communicate through ``ZZMI_MergedSkeleton.json`` in the workspace.

How Zenless Zone Zero poses a character
---------------------------------------
Every component (one IB) is skinned by its own stream-output draw before it is
rendered.  That draw binds the component's bone palette at ``vs-t0`` (a
``tbuffer`` of ``float4x3`` rows, three ``float4`` per bone) and the vertex
shader reads ``bones[BLENDINDICES]`` directly.  The game uploads each palette
with ``Map(WRITE_DISCARD)`` right before the component's own skinning draw.

Each palette is expressed in the space of the component's root bone, and the
indexed draws move the skinned vertices to world space with the component's
own object matrix.  The same bone therefore has a different matrix in every
component that uses it.  Two components that share a bone, however, agree on
it in world space, and that shared bone ("anchor") gives the conversion
between their spaces without any data from the indexed draws:

    E(parent <- child) = P_parent[anchor] * inverse(P_child[anchor])

The merged skeleton keeps every merged component's palette in one pool, at the
component's ``vg_offset``.  Before a component is skinned, a palette in that
component's own space is built from the pool (bones of other components are
converted through the anchor chain), so a mesh drawn through any component can
use the bones of every merged component.

Face meshes are driven by shape keys (a compute pass rewrites their position
buffer every frame), so they are not merged; the exporter keeps the game's own
mesh for them, like EFMI does for CPU-posed components.  Components that share
no bone with the merged ones cannot be converted and stay standalone.

Uniform one-frame delay (optional export)
-----------------------------------------
By default a component skinned before another one reads that component's bones
from the previous frame.  The delayed export instead skins the whole character
with the bones of the previous frame, as WWMI's merged skeleton does: the pool
is copied at ``[Present]`` and every palette is built from that copy.  Faces
and standalone components import their own palette into extra pool ranges
after the merged bones and are skinned with their previous palette as well;
their vertex buffers (the faces' expressions) stay the game's current ones.

That delays every part relative to its own root, and the game places each part
with the current matrix of that root (vs-cb1); the roots are bones (hips,
chest, head), so every part is also drawn with its object matrix of the
previous frame (``PLACE_SHADER_SOURCE``), moved by how far the character moved
since then: the whole character is drawn in the pose of the previous frame, at
its current position.  The distance is that of the first part drawn in the
frame and every part is moved by it, so the parts stay together.  (Drawn where
it was, the character trails by its movement of one frame, which changes with
the frame time: at flying speed it shakes against the world.)

Same-frame skinning (optional export)
-------------------------------------
The game skins every component before it draws the first one.  The same-frame
export therefore does not skin a merged part at its own stream-output draw:
the part only imports its bones there, and at its first draw of the frame,
when every component has uploaded its bones, its palette is built from the
pool and ``SKIN_SHADER_SOURCE`` skins it into a buffer of the mod, which is
bound as the part's vertex buffer (vb0) for its draws; the result of the
previous frame is bound as vb3 (previous positions for motion vectors).  The
whole character uses the bones of the current frame.

The character can be on screen more than once (the copies are skinned
interleaved, a few metres or less apart).  Nothing in a skinning draw tells
the copies apart except its stream-output buffer: each copy of a component
skins into its own two buffers, which the game alternates.  The ini gives each
buffer a slot of the instance store by its identity (``@so0``, ``@vb0``), the
bones are imported into that slot, and at its draws a component records its
root's world matrix (vs-cb1) for its slots.  When a copy of a merged part is
drawn, ``INSTANCE_MATCH_SHADER_SOURCE`` picks, starting from that part, the
instance of every component whose recorded matrix puts the shared bone where
the copy puts it; the bones are gathered into the pool and the copy is skinned
into a buffer of its own (one per stream-output buffer, so vb3 is the copy's
result of the previous frame).
"""

from __future__ import annotations

import json
import os

import numpy


PROFILE_FILENAME = "ZZMI_MergedSkeleton.json"
PROFILE_MODE = "ZZMI_MERGED_SKELETON"
PROFILE_FORMAT_VERSION = 1
SUBMESH_METADATA_KEY = "ZZMIMergedSkeleton"

ROLE_MERGED = "merged"
ROLE_SHAPE_KEYED = "shape_keyed"
ROLE_STANDALONE = "standalone"
ROLES = (ROLE_MERGED, ROLE_SHAPE_KEYED, ROLE_STANDALONE)

OBJECT_MARKER_MERGED_SKELETON = "LoyalTools:ZZMIMergedSkeleton"
OBJECT_MARKER_COMPONENT_ID = "LoyalTools:ZZMIComponentId"
OBJECT_MARKER_ROLE = "LoyalTools:ZZMIRole"
TREE_MARKER_MERGED_SKELETON = "LoyalTools:ZZMIMergedSkeleton"

# The game's skinning shaders declare ``float4x3 bones[1024]``.
MAX_POOL_BONES = 1024
NO_PARENT = -1

# World space matching of bone matrices.  Matching bones agree to ~2e-5 in the
# captured frames (float32 precision of world coordinates); distinct bones
# differ by far more.  The translation tolerance grows with the distance from
# the world origin because the inputs are float32.
MATCH_ROTATION_TOLERANCE = 1e-4
MATCH_TRANSLATION_TOLERANCE = 1e-3
MATCH_TRANSLATION_RELATIVE_TOLERANCE = 4e-6


class ZZMIMergedSkeletonError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Profile contract
# ---------------------------------------------------------------------------

def get_profile_path(workspace_folder: str) -> str:
    return os.path.join(os.path.abspath(str(workspace_folder)), PROFILE_FILENAME)


def _as_int(value, field_name: str, minimum: int | None = 0) -> int:
    if isinstance(value, bool):
        raise ZZMIMergedSkeletonError(field_name + " 必须是整数。")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ZZMIMergedSkeletonError(field_name + " 必须是整数。")
    if minimum is not None and result < minimum:
        raise ZZMIMergedSkeletonError(field_name + " 不能小于 " + str(minimum) + "。")
    return result


def normalize_vg_map(vg_map, field_name: str = "vg_map") -> dict[str, int]:
    if vg_map is None:
        return {}
    if not isinstance(vg_map, dict):
        raise ZZMIMergedSkeletonError(field_name + " 必须是对象映射。")
    normalized = {}
    for local_id, global_id in vg_map.items():
        normalized[str(_as_int(local_id, field_name + " 的本地骨骼"))] = _as_int(
            global_id, field_name + " 的全局骨骼"
        )
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict):
        raise ZZMIMergedSkeletonError("骨骼合并配置根节点必须是 JSON 对象。")
    if profile.get("mode") != PROFILE_MODE:
        raise ZZMIMergedSkeletonError("骨骼合并配置 mode 无效，期望 " + PROFILE_MODE + "。")
    if _as_int(profile.get("format_version", 0), "format_version") != PROFILE_FORMAT_VERSION:
        raise ZZMIMergedSkeletonError(
            "不支持的骨骼合并配置版本: " + str(profile.get("format_version"))
        )
    raw_components = profile.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ZZMIMergedSkeletonError("骨骼合并配置没有 components。")

    components = []
    seen_ib = set()
    seen_unique_strs = set()
    for list_index, raw in enumerate(raw_components):
        if not isinstance(raw, dict):
            raise ZZMIMergedSkeletonError("components[" + str(list_index) + "] 必须是对象。")
        component = dict(raw)
        label = "components[" + str(list_index) + "]"
        if _as_int(component.get("component_id", -1), label + ".component_id") != list_index:
            raise ZZMIMergedSkeletonError("component_id 必须从 0 连续编号。")
        component["component_id"] = list_index
        component["ib_hash"] = str(component.get("ib_hash", "")).strip().lower()
        if not component["ib_hash"]:
            raise ZZMIMergedSkeletonError(label + " 缺少 ib_hash。")
        if component["ib_hash"] in seen_ib:
            raise ZZMIMergedSkeletonError("ib_hash 重复: " + component["ib_hash"])
        seen_ib.add(component["ib_hash"])
        label = component["ib_hash"]

        unique_strs = component.get("unique_strs")
        if not isinstance(unique_strs, list) or not unique_strs:
            raise ZZMIMergedSkeletonError(label + " 缺少 unique_strs。")
        component["unique_strs"] = [str(item).strip() for item in unique_strs]
        for unique_str in component["unique_strs"]:
            if not unique_str.startswith(component["ib_hash"] + "-"):
                raise ZZMIMergedSkeletonError(label + " 的子网格 " + unique_str + " 不属于该 IB。")
            if unique_str in seen_unique_strs:
                raise ZZMIMergedSkeletonError("unique_str 重复: " + unique_str)
            seen_unique_strs.add(unique_str)

        component["role"] = str(component.get("role", ""))
        if component["role"] not in ROLES:
            raise ZZMIMergedSkeletonError(label + " 的 role 无效: " + component["role"])
        component["reason"] = str(component.get("reason", ""))
        component["blend_vb"] = str(component.get("blend_vb", "")).strip().lower()
        component["pose_vs"] = str(component.get("pose_vs", "")).strip().lower()
        # Position / Blend element layout of the skinning draw (for diagnostics).
        component["layout_signature"] = str(component.get("layout_signature", ""))
        component["pose_order"] = _as_int(component.get("pose_order", 0), label + ".pose_order")
        component["vertex_count"] = _as_int(component.get("vertex_count", 0), label + ".vertex_count")
        component["indices_per_vertex"] = _as_int(
            component.get("indices_per_vertex", 4), label + ".indices_per_vertex", 1
        )
        # VertexCount of the game's skinning Draw; 0 in profiles written before it was stored.
        component["pose_vertex_count"] = _as_int(
            component.get("pose_vertex_count", 0), label + ".pose_vertex_count"
        )
        component["palette_bone_count"] = _as_int(
            component.get("palette_bone_count", 0), label + ".palette_bone_count"
        )

        if component["role"] == ROLE_MERGED:
            component["vg_offset"] = _as_int(component.get("vg_offset", 0), label + ".vg_offset")
            component["vg_count"] = _as_int(component.get("vg_count", 0), label + ".vg_count", 1)
            component["vg_map"] = normalize_vg_map(component.get("vg_map"), label + ".vg_map")
            missing = [
                local_id for local_id in range(component["vg_count"])
                if str(local_id) not in component["vg_map"]
            ]
            if missing:
                raise ZZMIMergedSkeletonError(
                    label + " 的 vg_map 不完整，缺少本地骨骼: " + ", ".join(map(str, missing[:8]))
                )
            component["parent_component_id"] = _as_int(
                component.get("parent_component_id", NO_PARENT), label + ".parent_component_id"
            )
            component["anchor_parent_slot"] = _as_int(
                component.get("anchor_parent_slot", 0), label + ".anchor_parent_slot"
            )
            component["anchor_self_slot"] = _as_int(
                component.get("anchor_self_slot", 0), label + ".anchor_self_slot"
            )
        else:
            component["vg_offset"] = 0
            component["vg_count"] = 0
            component["vg_map"] = {}
            component["parent_component_id"] = NO_PARENT
            component["anchor_parent_slot"] = 0
            component["anchor_self_slot"] = 0
        components.append(component)

    merged = [component for component in components if component["role"] == ROLE_MERGED]
    if not merged:
        raise ZZMIMergedSkeletonError("骨骼合并配置里没有参与合并的部件。")
    expected_offset = 0
    for component in merged:
        if component["vg_offset"] != expected_offset:
            raise ZZMIMergedSkeletonError(
                component["ib_hash"] + " 的 vg_offset 应为 " + str(expected_offset)
                + "，实际为 " + str(component["vg_offset"]) + "。"
            )
        expected_offset += component["vg_count"]
    bones_count = expected_offset
    if bones_count > MAX_POOL_BONES:
        raise ZZMIMergedSkeletonError(
            "合并后的骨骼数 " + str(bones_count) + " 超过游戏蒙皮着色器上限 " + str(MAX_POOL_BONES) + "。"
        )

    by_id = {component["component_id"]: component for component in components}
    roots = [component for component in merged if component["parent_component_id"] == component["component_id"]]
    if len(roots) != 1:
        raise ZZMIMergedSkeletonError("骨骼合并配置必须正好有一个根部件。")
    for component in merged:
        parent = by_id.get(component["parent_component_id"])
        if parent is None or parent["role"] != ROLE_MERGED:
            raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的锚点父部件无效。")
        for slot_key in ("anchor_parent_slot", "anchor_self_slot"):
            if component[slot_key] >= bones_count:
                raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的 " + slot_key + " 超出骨骼池。")
        if parent is not component:
            if not _slot_in(parent, component["anchor_parent_slot"]):
                raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的锚点不在父部件的骨骼范围内。")
            if not _slot_in(component, component["anchor_self_slot"]):
                raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的锚点不在自身骨骼范围内。")
        for global_id in component["vg_map"].values():
            if global_id >= bones_count:
                raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的 vg_map 超出骨骼池。")
        # The chain must end at the root without cycles.
        visited = set()
        current = component
        while current["parent_component_id"] != current["component_id"]:
            if current["component_id"] in visited:
                raise ZZMIMergedSkeletonError(component["ib_hash"] + " 的锚点链有环。")
            visited.add(current["component_id"])
            current = by_id[current["parent_component_id"]]

    result = dict(profile)
    result["format_version"] = PROFILE_FORMAT_VERSION
    result["mode"] = PROFILE_MODE
    result["components"] = components
    result["bones_count"] = bones_count
    result["root_component_id"] = roots[0]["component_id"]
    result["source_frame_dump"] = str(result.get("source_frame_dump", "")).strip()
    return result


def _slot_in(component: dict, slot: int) -> bool:
    return component["vg_offset"] <= slot < component["vg_offset"] + component["vg_count"]


def load_profile(workspace_folder: str, required: bool = True) -> dict | None:
    path = get_profile_path(workspace_folder)
    if not os.path.isfile(path):
        if required:
            raise ZZMIMergedSkeletonError(
                "当前工作空间没有 " + PROFILE_FILENAME + "，请先在绝区零预设下用“骨骼合并”提取。"
            )
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            profile = json.load(file)
    except Exception as exc:
        raise ZZMIMergedSkeletonError("读取骨骼合并配置失败: " + repr(exc))
    return validate_profile(profile)


def write_profile(workspace_folder: str, profile: dict) -> str:
    normalized = validate_profile(profile)
    path = get_profile_path(workspace_folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=4)
    return path


def get_component_by_ib(profile: dict) -> dict[str, dict]:
    return {component["ib_hash"]: component for component in profile["components"]}


def get_component_by_unique_str(profile: dict) -> dict[str, dict]:
    return {
        unique_str: component
        for component in profile["components"]
        for unique_str in component["unique_strs"]
    }


def merged_components(profile: dict) -> list[dict]:
    return [component for component in profile["components"] if component["role"] == ROLE_MERGED]


def make_submesh_metadata(component: dict) -> dict:
    """Small block written into every part's SubmeshJson."""
    return {
        "Profile": PROFILE_FILENAME,
        "ComponentId": int(component["component_id"]),
        "Role": component["role"],
        "VGOffset": int(component.get("vg_offset", 0)),
        "VGCount": int(component.get("vg_count", 0)),
    }


def parse_submesh_metadata(json_dict: dict) -> dict | None:
    raw = json_dict.get(SUBMESH_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("Role", ""))
    if role not in ROLES:
        raise ZZMIMergedSkeletonError("子网格的骨骼合并角色无效: " + role)
    return {
        "profile": str(raw.get("Profile", PROFILE_FILENAME)),
        "component_id": _as_int(raw.get("ComponentId", 0), "ComponentId"),
        "role": role,
    }


# ---------------------------------------------------------------------------
# Export: canonical ids -> pool slots
# ---------------------------------------------------------------------------

def build_private_remap(component: dict) -> dict[int, int]:
    """Map canonical global ids to this component's own copy of the bone.

    Blender keeps one vertex group per canonical bone, so weights move freely
    between components.  A bone the drawing component has itself is read from
    its own palette (current in the frame, and copied exactly); only bones it
    does not have keep the canonical slot of another component.
    """
    vg_offset = int(component["vg_offset"])
    local_ids_by_canonical: dict[int, list[int]] = {}
    for local_text, canonical in component["vg_map"].items():
        local_ids_by_canonical.setdefault(int(canonical), []).append(int(local_text))
    remap = {}
    for canonical, local_ids in local_ids_by_canonical.items():
        own = canonical - vg_offset
        remap[canonical] = vg_offset + (own if own in local_ids else min(local_ids))
    return remap


def slot_owner_ids(profile: dict, slots) -> set[int]:
    """Component ids whose pool range contains any of ``slots``."""
    owners = set()
    ranges = [
        (component["vg_offset"], component["vg_offset"] + component["vg_count"], component["component_id"])
        for component in merged_components(profile)
    ]
    for slot in {int(value) for value in slots}:
        for start, end, component_id in ranges:
            if start <= slot < end:
                owners.add(component_id)
                break
        else:
            raise ZZMIMergedSkeletonError("骨骼编号 " + str(slot) + " 不属于任何合并部件。")
    return owners


def anchor_chain_ids(profile: dict, component_id: int) -> list[int]:
    """``component_id`` followed by its anchor ancestors up to the root."""
    by_id = {component["component_id"]: component for component in profile["components"]}
    chain = [int(component_id)]
    current = by_id[int(component_id)]
    while current["parent_component_id"] != current["component_id"]:
        current = by_id[current["parent_component_id"]]
        chain.append(current["component_id"])
    return chain


def palette_dependencies(profile: dict, component_id: int, foreign_slots) -> set[int]:
    """Components whose palettes a merged palette of ``component_id`` reads.

    That is the owners of the foreign bones, and every component on the anchor
    chains of the drawing component and of those owners.
    """
    dependencies = set(anchor_chain_ids(profile, component_id))
    for owner_id in slot_owner_ids(profile, foreign_slots):
        dependencies.update(anchor_chain_ids(profile, owner_id))
    dependencies.discard(int(component_id))
    return dependencies


# ---------------------------------------------------------------------------
# Runtime tables (written inline into the mod ini)
# ---------------------------------------------------------------------------

def delayed_pool_layout(profile: dict, component_ids) -> list[dict]:
    """Pool ranges for components that are not merged but are delayed too.

    Faces and standalone components keep their own bones; with the uniform
    one-frame delay they import their palette into a range after the merged
    bones so their previous palette can be read back.  Returns one entry per
    component, in the given order: ``component_id``, ``offset``, ``count`` and
    ``table_index`` (their entries follow the merged ones in the runtime table).
    """
    by_id = {component["component_id"]: component for component in profile["components"]}
    merged_count = len(merged_components(profile))
    offset = int(profile["bones_count"])
    layout = []
    for position, component_id in enumerate(component_ids):
        component = by_id[int(component_id)]
        if component["role"] == ROLE_MERGED:
            raise ZZMIMergedSkeletonError("部件 " + str(component_id) + " 已经在骨骼池里。")
        count = int(component["palette_bone_count"])
        if count <= 0:
            raise ZZMIMergedSkeletonError(component["ib_hash"] + " 没有记录骨骼表大小，请重新提取。")
        layout.append({
            "component_id": int(component_id), "offset": offset, "count": count,
            "table_index": merged_count + position,
        })
        offset += count
    return layout


def runtime_component_table(profile: dict, delayed_layout=()) -> list[int]:
    """Two ``uint4`` per merged component, in pool order.

    [2k + 0] = (vg_offset, vg_count, parent index, 1)
    [2k + 1] = (anchor slot in the parent, anchor slot in this component, 0, 0)

    Indices are positions in this table, not profile component ids.  Entries
    of ``delayed_layout`` (see ``delayed_pool_layout``) follow as roots of
    their own: (offset, count, own index, 1), (0, 0, 0, 0).
    """
    merged = merged_components(profile)
    index_by_id = {component["component_id"]: index for index, component in enumerate(merged)}
    values = []
    for component in merged:
        values += [
            component["vg_offset"], component["vg_count"],
            index_by_id[component["parent_component_id"]], 1,
            component["anchor_parent_slot"], component["anchor_self_slot"], 0, 0,
        ]
    for entry in delayed_layout:
        values += [entry["offset"], entry["count"], entry["table_index"], 1, 0, 0, 0, 0]
    return values


def runtime_component_index(profile: dict, component_id: int) -> int:
    for index, component in enumerate(merged_components(profile)):
        if component["component_id"] == int(component_id):
            return index
    raise ZZMIMergedSkeletonError("部件 " + str(component_id) + " 不参与骨骼合并。")


def runtime_params(profile: dict, component_id: int, delayed_layout=()) -> list[int]:
    """``uint4`` (table index, merged component count, pool bone count, 0)."""
    table_index = None
    for entry in delayed_layout:
        if entry["component_id"] == int(component_id):
            table_index = entry["table_index"]
    if table_index is None:
        table_index = runtime_component_index(profile, component_id)
    return [
        table_index,
        len(merged_components(profile)),
        int(profile["bones_count"]),
        0,
    ]


IMPORT_SHADER_FILENAME = "MergedSkeleton_Import.hlsl"
PALETTE_SHADER_FILENAME = "MergedSkeleton_Palette.hlsl"
DELAY_SHADER_FILENAME = "MergedSkeleton_Delay.hlsl"
PLACE_SHADER_FILENAME = "MergedSkeleton_Place.hlsl"
SKIN_SHADER_FILENAME = "MergedSkeleton_Skin.hlsl"
INSTANCE_IMPORT_SHADER_FILENAME = "MergedSkeleton_InstanceImport.hlsl"
INSTANCE_RECORD_SHADER_FILENAME = "MergedSkeleton_InstanceRecord.hlsl"
INSTANCE_MATCH_SHADER_FILENAME = "MergedSkeleton_InstanceMatch.hlsl"
INSTANCE_GATHER_SHADER_FILENAME = "MergedSkeleton_InstanceGather.hlsl"

# Same-frame skinning with several copies of the character on screen: copies of one
# component are told apart by their stream-output buffers (two per copy, the game
# alternates them), each buffer gets a slot of the instance store.
MAX_INSTANCES = 8
MAX_INSTANCE_COMPONENTS = 32
INSTANCE_RECORD_ROWS = 5
INSTANCE_EXPIRATION_FRAMES = 600
# Metres between where a candidate and the matched component put a shared bone (the matcher's
# MATCH_DISTANCE): within it a candidate is taken even when another one has no record yet.
MATCH_DISTANCE = 0.25

# Vertices the same-frame skinning reads and writes: POSITION float3, NORMAL float3,
# TANGENT float4, the layout of the game's skinning input and stream output.
SKIN_VERTEX_STRIDE = 40
SKIN_POSITION_LAYOUT = (("POSITION", "R32G32B32_FLOAT"), ("NORMAL", "R32G32B32_FLOAT"),
                        ("TANGENT", "R32G32B32A32_FLOAT"))
# Weight formats the skinning shader decodes (code in its parameters).
SKIN_WEIGHT_FORMATS = {"R32G32B32A32_FLOAT": 1, "R16G16B16A16_FLOAT": 2, "R16G16B16A16_UNORM": 3,
                       "R8G8B8A8_UNORM": 4}
SKIN_FOUR_INDEX_FORMATS = ("R32G32B32A32_UINT", "R32G32B32A32_SINT")
SKIN_ONE_INDEX_FORMATS = ("R32_UINT", "R32_SINT")

# The game's per-draw object constants (vs-cb1, 29 registers): rows 0-3 the
# object-to-world matrix (row vectors), 4-7 its inverse, 20-23 the matrix of
# the previous frame.
OBJECT_CB_REGISTERS = 29

IMPORT_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: bone import.
// Copies the bone palette bound for this component's skinning draw (vs-t0)
// into the component's range of the merged bone pool.

// Read the palette the way the game's skinning shader does (a tbuffer), so
// the view format of the game's buffer does not matter.
tbuffer ComponentBones : register(t0)
{
    float4 ComponentBoneRows[3072];
};

// Two entries per merged component:
//   [2k + 0] = (vg_offset, vg_count, parent index, 1)
//   [2k + 1] = (anchor slot in parent, anchor slot in self, 0, 0)
Buffer<uint4> Components : register(t1);

// (table index of this component, merged component count, pool bone count, 0)
Buffer<uint4> Params : register(t2);

RWBuffer<float4> Pool : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    uint4 component = Components[params.x * 2];
    uint bone = thread_id.x;
    if (bone >= component.y)
        return;

    uint src = bone * 3;
    uint dst = (component.x + bone) * 3;
    Pool[dst + 0] = ComponentBoneRows[src + 0];
    Pool[dst + 1] = ComponentBoneRows[src + 1];
    Pool[dst + 2] = ComponentBoneRows[src + 2];
}
"""

PALETTE_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: palette builder.
// Builds the bone palette for one component's skinning draw.  Every pool slot
// is expressed in the space of that component's root bone; bones of other
// components are converted through bones the components share (anchors):
//     E(parent <- child) = Pool[anchor in parent] * inverse(Pool[anchor in child])

Buffer<float4> Pool : register(t0);

// Two entries per merged component:
//   [2k + 0] = (vg_offset, vg_count, parent index, 1)
//   [2k + 1] = (anchor slot in parent, anchor slot in self, 0, 0)
Buffer<uint4> Components : register(t1);

// (table index of this component, merged component count, pool bone count, 0)
Buffer<uint4> Params : register(t2);

RWBuffer<float4> Palette : register(u0);

// Affine 3x4 matrix stored as rows, as the game's float4x3 bones are.
struct Affine
{
    float4 r0;
    float4 r1;
    float4 r2;
};

Affine Identity()
{
    Affine m;
    m.r0 = float4(1, 0, 0, 0);
    m.r1 = float4(0, 1, 0, 0);
    m.r2 = float4(0, 0, 1, 0);
    return m;
}

Affine LoadSlot(uint slot)
{
    Affine m;
    m.r0 = Pool[slot * 3 + 0];
    m.r1 = Pool[slot * 3 + 1];
    m.r2 = Pool[slot * 3 + 2];
    return m;
}

// a * b, both with an implicit (0, 0, 0, 1) bottom row.
float4 MulRow(float4 row, Affine b)
{
    return row.x * b.r0 + row.y * b.r1 + row.z * b.r2 + float4(0, 0, 0, row.w);
}

Affine Mul(Affine a, Affine b)
{
    Affine m;
    m.r0 = MulRow(a.r0, b);
    m.r1 = MulRow(a.r1, b);
    m.r2 = MulRow(a.r2, b);
    return m;
}

Affine Inverse(Affine m)
{
    float3 a = m.r0.xyz;
    float3 b = m.r1.xyz;
    float3 c = m.r2.xyz;
    float3 c0 = cross(b, c);
    float3 c1 = cross(c, a);
    float3 c2 = cross(a, b);
    float det = dot(a, c0);
    // Slots that were never imported are zero; keep the result finite.
    Affine r = Identity();
    if (abs(det) >= 1e-20)
    {
        float inv_det = 1.0 / det;
        // Rows of the inverse 3x3 are the columns of the cofactor matrix.
        float3 i0 = float3(c0.x, c1.x, c2.x) * inv_det;
        float3 i1 = float3(c0.y, c1.y, c2.y) * inv_det;
        float3 i2 = float3(c0.z, c1.z, c2.z) * inv_det;
        float3 t = float3(m.r0.w, m.r1.w, m.r2.w);
        r.r0 = float4(i0, -dot(i0, t));
        r.r1 = float4(i1, -dot(i1, t));
        r.r2 = float4(i2, -dot(i2, t));
    }
    return r;
}

// Conversion from the space of component k to the space of the root.
Affine ToRoot(uint k)
{
    Affine acc = Identity();
    [loop]
    for (uint step = 0; step < 32; ++step)
    {
        uint4 entry = Components[k * 2 + 0];
        if (entry.z == k)
            break;
        uint4 anchor = Components[k * 2 + 1];
        Affine edge = Mul(LoadSlot(anchor.x), Inverse(LoadSlot(anchor.y)));
        acc = Mul(edge, acc);
        k = entry.z;
    }
    return acc;
}

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    uint slot = thread_id.x;
    if (slot >= params.z)
        return;

    uint owner = params.y;
    [loop]
    for (uint k = 0; k < params.y; ++k)
    {
        uint4 entry = Components[k * 2 + 0];
        if (slot >= entry.x && slot < entry.x + entry.y)
        {
            owner = k;
            break;
        }
    }

    Affine bone = LoadSlot(slot);
    if (owner != params.x && owner < params.y)
    {
        Affine to_self = Mul(Inverse(ToRoot(params.x)), ToRoot(owner));
        bone = Mul(to_self, bone);
    }
    Palette[slot * 3 + 0] = bone.r0;
    Palette[slot * 3 + 1] = bone.r1;
    Palette[slot * 3 + 2] = bone.r2;
}
"""

DELAY_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: uniform one-frame delay.
// Copies a range of bones from one pool to the start of another buffer: at
// [Present] the whole pool into the previous-frame pool, and before a
// component's skinning draw its own bones of the previous frame into its
// palette.

Buffer<float4> Source : register(t0);

// (first bone in Source, bone count, 0, 0)
Buffer<uint4> Params : register(t1);

RWBuffer<float4> Target : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    uint bone = thread_id.x;
    if (bone >= params.y)
        return;

    uint src = (params.x + bone) * 3;
    uint dst = bone * 3;
    Target[dst + 0] = Source[src + 0];
    Target[dst + 1] = Source[src + 1];
    Target[dst + 2] = Source[src + 2];
}
"""

PLACE_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: uniform one-frame delay, placement.
// Every part is skinned with the bones of the previous frame, in the space of
// its own root, and the game places each part with that root's current matrix
// (the roots are bones: hips, chest, head).  So every part is also drawn with
// its object matrix of the previous frame, moved by how far the character
// moved since then: the pose of the previous frame at the current position.
// That distance is the first drawn part's of the frame, and every part and
// pass of the frame is moved by it, so the parts stay together.
// Only rows 0-3 of the game's object constants (the current matrix) are
// written for every pass; the other rows can hold what an earlier draw left,
// so the matrices of earlier frames come from this mod's own records.
// The object constants (29 registers) are copied with rows 0-3 = the matrix
// drawn with, 4-7 its inverse and 20-23 the matrix drawn with in the previous
// frame (motion vectors).

cbuffer SelfObject : register(b0)
{
    float4 SelfRows[29];
};

// This part's records of the previous frame: rows 0-3 its matrix, 4-7 the
// matrix it was drawn with.  Rows 3 and 7 have w = 0 until they are written.
Buffer<float4> History : register(t0);

// (1 = place, 0 = only record, 2 = clear Shift (at [Present]), 0)
Buffer<uint4> Params : register(t1);

RWBuffer<float4> Output : register(u0);
// This frame's records: rows 0-3 the current matrix, 4-7 the one drawn with.
RWBuffer<float4> Record : register(u1);
// Shared by every part: the distance of this frame (0-2) and 1 once it is set
// (3).  One float per element, the format every device can read from a UAV.
RWBuffer<float> Shift : register(u2);

float4x4 Rows(float4 r0, float4 r1, float4 r2, float4 r3)
{
    return float4x4(r0, r1, r2, r3);
}

bool IsPlacement(float4x4 m)
{
    float3 w = abs(float3(m[0].w, m[1].w, m[2].w));
    float det = determinant((float3x3)m);
    return all(w < 1e-3) && abs(m[3].w - 1.0) < 1e-3 && det > 0.25 && det < 4.0
        && all(abs(m[3].xyz) < 1e6);
}

// Inverse of an affine matrix in row-vector form.
float4x4 InverseAffine(float4x4 m)
{
    float3 a = m[0].xyz;
    float3 b = m[1].xyz;
    float3 c = m[2].xyz;
    float3 c0 = cross(b, c);
    float3 c1 = cross(c, a);
    float3 c2 = cross(a, b);
    float inv_det = 1.0 / dot(a, c0);
    float3x3 r = float3x3(
        float3(c0.x, c1.x, c2.x) * inv_det,
        float3(c0.y, c1.y, c2.y) * inv_det,
        float3(c0.z, c1.z, c2.z) * inv_det);
    float3 t = -mul(m[3].xyz, r);
    return float4x4(float4(r[0], 0), float4(r[1], 0), float4(r[2], 0), float4(t, 1));
}

[numthreads(1, 1, 1)]
void main()
{
    uint mode = Params[0].x;
    if (mode == 2)
    {
        Shift[0] = 0;
        Shift[1] = 0;
        Shift[2] = 0;
        Shift[3] = 0;
        return;
    }

    [unroll]
    for (uint i = 0; i < 29; ++i)
        Output[i] = SelfRows[i];

    float4x4 self_now = Rows(SelfRows[0], SelfRows[1], SelfRows[2], SelfRows[3]);
    if (!IsPlacement(self_now))
        return;
    float4x4 self_previous = Rows(History[0], History[1], History[2], History[3]);
    float4x4 placed_previous = Rows(History[4], History[5], History[6], History[7]);
    float4x4 placed = self_now;
    // Rows are stored one by one (looping over two buffers at once lost rows on WARP).
    if (mode == 1 && IsPlacement(self_previous))
    {
        float3 shift;
        if (Shift[3] == 1.0)
        {
            shift = float3(Shift[0], Shift[1], Shift[2]);
        }
        else
        {
            shift = self_now[3].xyz - self_previous[3].xyz;
            Shift[0] = shift.x;
            Shift[1] = shift.y;
            Shift[2] = shift.z;
            Shift[3] = 1.0;
        }
        placed = self_previous;
        placed[3].xyz += shift;
        float4x4 inverse = InverseAffine(placed);
        if (!IsPlacement(placed_previous))
            placed_previous = placed;
        Output[0] = placed[0];
        Output[1] = placed[1];
        Output[2] = placed[2];
        Output[3] = placed[3];
        Output[4] = inverse[0];
        Output[5] = inverse[1];
        Output[6] = inverse[2];
        Output[7] = inverse[3];
        Output[20] = placed_previous[0];
        Output[21] = placed_previous[1];
        Output[22] = placed_previous[2];
        Output[23] = placed_previous[3];
    }
    Record[0] = SelfRows[0];
    Record[1] = SelfRows[1];
    Record[2] = SelfRows[2];
    Record[3] = SelfRows[3];
    Record[4] = placed[0];
    Record[5] = placed[1];
    Record[6] = placed[2];
    Record[7] = placed[3];
}
"""


SKIN_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: same-frame skinning.
// The game skins each component with a stream-output draw right after uploading its bones,
// so a part that also uses bones of components skinned later in the frame would get their
// bones of the previous frame.  The game skins every component before it draws the first
// one, so the merged part is skinned here instead, at its first draw of the frame, into a
// buffer of the mod that is bound as its vertex buffer (vb0).  Each copy of the character
// has two such buffers, one per stream-output buffer the game alternates, so the result of
// the previous frame is bound as vb3 (previous positions, motion vectors).
// The arithmetic is that of the game's skinning shaders (e8425f64cfb887cd, four weights;
// 9684c4091fc9e35a, one bone): the rows of the blended float4x3 bone transform the
// position, the normal and the tangent (not normalised); the tangent's w is kept.
// When the bones of some component of this copy were not found (a copy that has just
// appeared), every vertex is written at the origin: the copy is not drawn this frame.
// After the vertices, one more element says whether they are a valid result (first word 1).
// When the buffer bound as vb3 for this copy is not (the copy was hidden in the previous frame,
// or the buffer is new), this frame's result is written there too, so the motion vectors start
// from where the copy is instead of from the origin.

// The merged palette in this part's space (three rows per bone), built from this frame's bones.
Buffer<float4> Palette : register(t0);
// The exported vertices: POSITION float3, NORMAL float3, TANGENT float4 (40 bytes).
ByteAddressBuffer Position : register(t1);
// The exported weight buffer.
ByteAddressBuffer Blend : register(t2);
// (vertex count, weight buffer stride,
//  weight offset | weight format << 16 (0 none, 1 R32G32B32A32_FLOAT, 2 R16G16B16A16_FLOAT,
//                                       3 R16G16B16A16_UNORM, 4 R8G8B8A8_UNORM),
//  index offset | one bone << 16)
Buffer<uint4> Params : register(t3);
// Row 0: (1 when the bones of every component of this copy were found, 0, 0, 0).
Buffer<uint4> Choice : register(t4);

// Skinned vertices, the layout of the game's stream output (40 bytes), then the validity word.
RWByteAddressBuffer Skinned : register(u0);
// This copy's buffer bound as vb3 in this frame (not bound when it is the same as u0).
RWByteAddressBuffer Previous : register(u1);

float4 LoadWeights(uint address, uint format)
{
    if (format == 1)
        return asfloat(Blend.Load4(address));
    if (format == 2)
    {
        uint2 halves = Blend.Load2(address);
        return float4(f16tof32(halves.x), f16tof32(halves.x >> 16), f16tof32(halves.y), f16tof32(halves.y >> 16));
    }
    if (format == 3)
    {
        uint2 halves = Blend.Load2(address);
        return float4(halves.x & 0xFFFF, halves.x >> 16, halves.y & 0xFFFF, halves.y >> 16) / 65535.0;
    }
    uint bytes = Blend.Load(address);
    return float4(bytes & 0xFF, (bytes >> 8) & 0xFF, (bytes >> 16) & 0xFF, bytes >> 24) / 255.0;
}

// One row of the blended bone, summed in the order of the game's shader.
float4 BlendedRow(uint4 bones, float4 weights, uint row)
{
    float4 value = weights.y * Palette[bones.y + row];
    value = weights.x * Palette[bones.x + row] + value;
    value = weights.z * Palette[bones.z + row] + value;
    return weights.w * Palette[bones.w + row] + value;
}

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    uint vertex = thread_id.x;
    if (vertex >= params.x)
        return;

    uint target = vertex * 40;
    uint validity = params.x * 40;
    bool found = Choice[0].x != 0;
    if (vertex == 0)
        Skinned.Store(validity, found ? 1 : 0);
    if (!found)
    {
        Skinned.Store4(target, uint4(0, 0, 0, 0));
        Skinned.Store4(target + 16, uint4(0, 0, 0, 0));
        Skinned.Store2(target + 32, uint2(0, 0));
        return;
    }

    uint source = vertex * 40;
    float4 homogeneous = float4(asfloat(Position.Load3(source)), 1.0);
    float3 normal = asfloat(Position.Load3(source + 12));
    float4 tangent = asfloat(Position.Load4(source + 24));

    uint blend = vertex * params.y;
    uint index_address = blend + (params.w & 0xFFFF);
    float4 row0, row1, row2;
    if (params.w & 0x10000)
    {
        uint bone = Blend.Load(index_address) * 3;
        row0 = Palette[bone + 0];
        row1 = Palette[bone + 1];
        row2 = Palette[bone + 2];
    }
    else
    {
        uint4 bones = Blend.Load4(index_address) * 3;
        float4 weights = LoadWeights(blend + (params.z & 0xFFFF), params.z >> 16);
        row0 = BlendedRow(bones, weights, 0);
        row1 = BlendedRow(bones, weights, 1);
        row2 = BlendedRow(bones, weights, 2);
    }

    float3 position = float3(dot(homogeneous, row0), dot(homogeneous, row1), dot(homogeneous, row2));
    float3 skinned_normal = float3(dot(normal, row0.xyz), dot(normal, row1.xyz), dot(normal, row2.xyz));
    float3 skinned_tangent = float3(dot(tangent.xyz, row0.xyz), dot(tangent.xyz, row1.xyz),
                                    dot(tangent.xyz, row2.xyz));
    uint4 now0 = uint4(asuint(position), asuint(skinned_normal.x));
    uint4 now1 = uint4(asuint(skinned_normal.yz), asuint(skinned_tangent.xy));
    uint2 now2 = uint2(asuint(skinned_tangent.z), asuint(tangent.w));
    Skinned.Store4(target, now0);
    Skinned.Store4(target + 16, now1);
    Skinned.Store2(target + 32, now2);
    if (Previous.Load(validity) != 1)
    {
        Previous.Store4(target, now0);
        Previous.Store4(target + 16, now1);
        Previous.Store2(target + 32, now2);
    }
}
"""

INSTANCE_COMMON_HLSL = r"""
// Affine 3x4 matrix stored as rows (column-vector convention), as the game's float4x3 bones are.
struct Affine
{
    float4 r0;
    float4 r1;
    float4 r2;
};

float4 MulRow(float4 row, Affine b)
{
    return row.x * b.r0 + row.y * b.r1 + row.z * b.r2 + float4(0, 0, 0, row.w);
}

Affine Mul(Affine a, Affine b)
{
    Affine m;
    m.r0 = MulRow(a.r0, b);
    m.r1 = MulRow(a.r1, b);
    m.r2 = MulRow(a.r2, b);
    return m;
}

Affine Inverse(Affine m)
{
    float3 a = m.r0.xyz;
    float3 b = m.r1.xyz;
    float3 c = m.r2.xyz;
    float3 c0 = cross(b, c);
    float3 c1 = cross(c, a);
    float3 c2 = cross(a, b);
    float det = dot(a, c0);
    Affine r;
    r.r0 = float4(1, 0, 0, 0);
    r.r1 = float4(0, 1, 0, 0);
    r.r2 = float4(0, 0, 1, 0);
    if (abs(det) >= 1e-20)
    {
        float inv_det = 1.0 / det;
        float3 i0 = float3(c0.x, c1.x, c2.x) * inv_det;
        float3 i1 = float3(c0.y, c1.y, c2.y) * inv_det;
        float3 i2 = float3(c0.z, c1.z, c2.z) * inv_det;
        float3 t = float3(m.r0.w, m.r1.w, m.r2.w);
        r.r0 = float4(i0, -dot(i0, t));
        r.r1 = float4(i1, -dot(i1, t));
        r.r2 = float4(i2, -dot(i2, t));
    }
    return r;
}

// The object constants hold the root's object-to-world matrix as row vectors.
Affine RowsToWorld(float4 r0, float4 r1, float4 r2, float4 r3)
{
    Affine m;
    m.r0 = float4(r0.x, r1.x, r2.x, r3.x);
    m.r1 = float4(r0.y, r1.y, r2.y, r3.y);
    m.r2 = float4(r0.z, r1.z, r2.z, r3.z);
    return m;
}

float3 Apply(Affine m, float3 p)
{
    float4 h = float4(p, 1.0);
    return float3(dot(m.r0, h), dot(m.r1, h), dot(m.r2, h));
}
"""

INSTANCE_IMPORT_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: bone import, one store slot per component instance.
// When the character is on screen more than once, every copy skins each component into its
// own stream-output buffers.  Each such buffer has a slot of the instance store (the ini
// picks it by the identity of the buffer bound for the skinning draw), so the copies never
// overwrite each other's bones.
// Slot layout: row 0 = (table index of the component, frame number, 1, 0), then the
// component's own bones, three rows each.

tbuffer ComponentBones : register(t0)
{
    float4 ComponentBoneRows[3072];
};

// Two entries per merged component:
//   [2k + 0] = (vg_offset, vg_count, parent index, 1)
//   [2k + 1] = (anchor slot in parent, anchor slot in self, 0, 0)
Buffer<uint4> Components : register(t1);

// (table index of this component, merged component count, slot count, rows per slot)
Buffer<uint4> Params : register(t2);

// x0 = slot, y0 = frame number
Texture1D<float4> IniParams : register(t120);

RWBuffer<float4> Store : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    float4 ini = IniParams.Load(int2(0, 0));
    uint slot = (uint)ini.x;
    if (slot >= params.z)
        return;
    uint base = slot * params.w;
    uint4 component = Components[params.x * 2];
    uint bone = thread_id.x;
    if (bone == 0)
        Store[base] = float4(params.x, ini.y, 1, 0);
    if (bone >= component.y)
        return;

    uint src = bone * 3;
    uint dst = base + 1 + bone * 3;
    Store[dst + 0] = ComponentBoneRows[src + 0];
    Store[dst + 1] = ComponentBoneRows[src + 1];
    Store[dst + 2] = ComponentBoneRows[src + 2];
}
"""

INSTANCE_RECORD_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: where a component instance is.
// At the draws of a component, its object constants (vs-cb1 rows 0-3, of the part of the buffer
// the game bound: several objects can share one buffer) hold its root's world matrix.
// It is recorded for the slots of both stream-output buffers of the instance (this
// frame's and the previous frame's, the game alternates them), so the slot skinned in the
// next frame has a matrix of at most one frame before.
// Record layout: five rows per slot, the matrix rows 0-3, then (table index, frame number, 1, 0).

cbuffer Object : register(b0)
{
    float4 ObjectRows[4];
};

// (table index of this component, merged component count, slot count, rows per slot)
Buffer<uint4> Params : register(t0);

// x0 = slot of this frame's buffer, y0 = slot of the previous frame's buffer (or the same),
// z0 = frame number, w0 = table index
Texture1D<float4> IniParams : register(t120);

RWBuffer<float4> Records : register(u0);

bool IsPlacement(float4 r0, float4 r1, float4 r2, float4 r3)
{
    float det = determinant(float3x3(r0.xyz, r1.xyz, r2.xyz));
    return all(abs(float3(r0.w, r1.w, r2.w)) < 1e-3) && abs(r3.w - 1.0) < 1e-3
        && det > 0.25 && det < 4.0 && all(abs(r3.xyz) < 1e6);
}

void Write(uint slot, uint slots, float4 ini)
{
    if (slot >= slots)
        return;
    uint base = slot * 5;
    Records[base + 0] = ObjectRows[0];
    Records[base + 1] = ObjectRows[1];
    Records[base + 2] = ObjectRows[2];
    Records[base + 3] = ObjectRows[3];
    Records[base + 4] = float4(ini.w, ini.z, 1, 0);
}

[numthreads(1, 1, 1)]
void main()
{
    if (!IsPlacement(ObjectRows[0], ObjectRows[1], ObjectRows[2], ObjectRows[3]))
        return;
    float4 ini = IniParams.Load(int2(0, 0));
    uint slots = Params[0].z;
    Write((uint)ini.x, slots, ini);
    if ((uint)ini.y != (uint)ini.x)
        Write((uint)ini.y, slots, ini);
}
"""

INSTANCE_MATCH_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: which component instances belong to this copy.
// Starting from the drawn part (its root's world matrix is in the object constants of this
// draw), the components are reached through the bones they share (anchors).  Of the
// instances of a component skinned in this frame, the one whose recorded world matrix puts
// the shared bone where the already matched component puts it belongs to this copy: another
// copy stands somewhere else or in another pose, this one agrees to within its movement of a
// frame.  A candidate is taken only when that is certain: when it agrees to within
// MATCH_DISTANCE, or, every candidate having a record, when it agrees clearly better than the
// next one (a record of the previous frame at speed); a copy that has just appeared has no
// record yet and is not matched (it is not drawn this frame) rather than given another copy's
// bones.  With a single copy on screen, a component skinned only once this frame is taken
// even without a record.  The world matrix of a matched component is derived through the
// shared bone, so it is of this frame.
""" + INSTANCE_COMMON_HLSL + r"""
cbuffer HostObject : register(b0)
{
    float4 HostRows[4];
};

Buffer<float4> Store : register(t0);
Buffer<float4> Records : register(t1);

// Two entries per merged component:
//   [2k + 0] = (vg_offset, vg_count, parent index, 1)
//   [2k + 1] = (anchor slot in parent, anchor slot in self, 0, 0)
Buffer<uint4> Components : register(t2);

// (table index of the drawn part, merged component count, slot count, rows per slot)
Buffer<uint4> Params : register(t3);

// Per merged component: (1 when this part reads its bones, 0, 0, 0).
Buffer<uint4> Needed : register(t4);

// x0 = slot of the drawn part, y0 = frame number
Texture1D<float4> IniParams : register(t120);

// Row 0: (1 when every needed component was found, 0, 0, 0); row 1 + k: (slot, found, 0, 0).
RWBuffer<uint4> Choice : register(u0);

static const uint MAX_COMPONENTS = 32;
static const uint NO_SLOT = 0xFFFFFFFF;
// Metres between where a candidate and the matched component put a shared bone.
static const float MATCH_DISTANCE = 0.25;

Affine LoadBone(uint slot, uint rows, uint local)
{
    uint base = slot * rows + 1 + local * 3;
    Affine m;
    m.r0 = Store[base + 0];
    m.r1 = Store[base + 1];
    m.r2 = Store[base + 2];
    return m;
}

// Where a bone puts the origin of the bind pose.
float3 Point(Affine bone)
{
    return float3(bone.r0.w, bone.r1.w, bone.r2.w);
}

[numthreads(1, 1, 1)]
void main()
{
    uint4 params = Params[0];
    float4 ini = IniParams.Load(int2(0, 0));
    uint host = params.x;
    uint count = min(params.y, MAX_COMPONENTS);
    uint slots = params.z;
    uint rows = params.w;
    uint host_slot = (uint)ini.x;
    float frame = ini.y;

    uint slot_of[MAX_COMPONENTS];
    Affine world[MAX_COMPONENTS];
    [loop]
    for (uint k = 0; k < MAX_COMPONENTS; ++k)
    {
        slot_of[k] = NO_SLOT;
        world[k].r0 = float4(1, 0, 0, 0);
        world[k].r1 = float4(0, 1, 0, 0);
        world[k].r2 = float4(0, 0, 1, 0);
    }

    bool host_found = host < count && host_slot < slots
        && Store[host_slot * rows].x == (float)host && Store[host_slot * rows].y == frame;
    // Several copies of the drawn part skinned this frame: several copies on screen.
    uint copies = 0;
    [loop]
    for (uint h = 0; h < slots; ++h)
    {
        float4 host_header = Store[h * rows];
        if (host_header.x == (float)host && host_header.y == frame)
            copies += 1;
    }
    if (host_found)
    {
        slot_of[host] = host_slot;
        world[host] = RowsToWorld(HostRows[0], HostRows[1], HostRows[2], HostRows[3]);
    }

    [loop]
    for (uint round = 0; round < count && host_found; ++round)
    {
        bool progress = false;
        [loop]
        for (uint c = 0; c < count; ++c)
        {
            uint p = Components[c * 2].z;
            if (p == c || p >= count)
                continue;
            uint4 anchor = Components[c * 2 + 1];
            uint known = 0, target = 0, known_anchor = 0, target_anchor = 0;
            if (slot_of[p] != NO_SLOT && slot_of[c] == NO_SLOT)
            {
                known = p;
                target = c;
                known_anchor = anchor.x;
                target_anchor = anchor.y;
            }
            else if (slot_of[c] != NO_SLOT && slot_of[p] == NO_SLOT)
            {
                known = c;
                target = p;
                known_anchor = anchor.y;
                target_anchor = anchor.x;
            }
            else
            {
                continue;
            }
            if (Needed[target].x == 0)
                continue;

            uint target_local = target_anchor - Components[target * 2].x;
            Affine known_bone = LoadBone(slot_of[known], rows, known_anchor - Components[known * 2].x);
            float3 expected = Apply(world[known], Point(known_bone));

            uint best = NO_SLOT;
            uint only = NO_SLOT;
            float best_distance = 3.0e38;
            float second_distance = 3.0e38;
            uint candidates = 0;
            uint unrecorded = 0;
            [loop]
            for (uint s = 0; s < slots; ++s)
            {
                float4 header = Store[s * rows];
                if (header.x != (float)target || header.y != frame)
                    continue;
                candidates += 1;
                only = s;
                float4 record = Records[s * 5 + 4];
                if (record.x == (float)target && record.z == 1 && frame - record.y <= 2)
                {
                    Affine recorded = RowsToWorld(Records[s * 5], Records[s * 5 + 1], Records[s * 5 + 2], Records[s * 5 + 3]);
                    float distance = length(Apply(recorded, Point(LoadBone(s, rows, target_local))) - expected);
                    if (distance < best_distance)
                    {
                        second_distance = best_distance;
                        best_distance = distance;
                        best = s;
                    }
                    else if (distance < second_distance)
                    {
                        second_distance = distance;
                    }
                }
                else
                {
                    unrecorded += 1;
                }
            }
            uint chosen = NO_SLOT;
            if (candidates == 1 && copies <= 1)
                chosen = only;
            else if (best != NO_SLOT && best_distance <= MATCH_DISTANCE)
                chosen = best;
            else if (best != NO_SLOT && unrecorded == 0 && candidates > 1 && best_distance * 2.0 < second_distance)
                chosen = best;
            if (chosen == NO_SLOT)
                continue;
            slot_of[target] = chosen;
            world[target] = Mul(Mul(world[known], known_bone), Inverse(LoadBone(chosen, rows, target_local)));
            progress = true;
        }
        if (!progress)
            break;
    }

    bool all_found = host_found;
    [loop]
    for (uint m = 0; m < count; ++m)
    {
        bool found = slot_of[m] != NO_SLOT;
        if (Needed[m].x != 0 && !found)
            all_found = false;
        Choice[1 + m] = uint4(found ? slot_of[m] : 0, found ? 1 : 0, 0, 0);
    }
    Choice[0] = uint4(all_found ? 1 : 0, 0, 0, 0);
}
"""

INSTANCE_GATHER_SHADER_SOURCE = r"""// Zenless Zone Zero merged skeleton: the bone pool of one copy of the character.
// Copies every component's bones from the store slot the matcher picked for this copy to the
// component's range of the pool, which the palette builder reads.

Buffer<float4> Store : register(t0);

// Two entries per merged component:
//   [2k + 0] = (vg_offset, vg_count, parent index, 1)
//   [2k + 1] = (anchor slot in parent, anchor slot in self, 0, 0)
Buffer<uint4> Components : register(t1);

// (table index of the drawn part, merged component count, slot count, rows per slot)
Buffer<uint4> Params : register(t2);

// Row 1 + k: (slot of component k, found, 0, 0).
Buffer<uint4> Choice : register(t3);

RWBuffer<float4> Pool : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint4 params = Params[0];
    uint bone = thread_id.x;
    [loop]
    for (uint k = 0; k < params.y; ++k)
    {
        uint4 entry = Components[k * 2];
        if (bone < entry.x || bone >= entry.x + entry.y)
            continue;
        uint4 choice = Choice[1 + k];
        float4 r0 = float4(0, 0, 0, 0);
        float4 r1 = float4(0, 0, 0, 0);
        float4 r2 = float4(0, 0, 0, 0);
        if (choice.y != 0)
        {
            uint base = choice.x * params.w + 1 + (bone - entry.x) * 3;
            r0 = Store[base + 0];
            r1 = Store[base + 1];
            r2 = Store[base + 2];
        }
        Pool[bone * 3 + 0] = r0;
        Pool[bone * 3 + 1] = r1;
        Pool[bone * 3 + 2] = r2;
        return;
    }
}
"""


def same_frame_skin_layout(position_elements, blend_elements):
    """Weight buffer layout for the skinning shader, or (None, reason) when it cannot skin it.

    Elements are (semantic name, semantic index, format, byte width) in buffer order.
    Returns ((stride, weight offset, weight format code, index offset, one bone), None).
    """
    position = [(name, fmt) for name, index, fmt, _ in position_elements if index == 0]
    if tuple(position) != SKIN_POSITION_LAYOUT or sum(width for *_, width in position_elements) != SKIN_VERTEX_STRIDE:
        return None, "顶点不是 POSITION/NORMAL/TANGENT 的 32 位浮点布局"
    offset = 0
    weights = indices = None
    for name, index, fmt, width in blend_elements:
        if index == 0 and name == "BLENDINDICES":
            indices = (offset, fmt)
        elif index == 0 and name in ("BLENDWEIGHT", "BLENDWEIGHTS"):
            weights = (offset, fmt)
        offset += int(width)
    if indices is None:
        return None, "没有 BLENDINDICES"
    if indices[1] in SKIN_ONE_INDEX_FORMATS:
        return (offset, 0, 0, indices[0], True), None
    if indices[1] not in SKIN_FOUR_INDEX_FORMATS:
        return None, "BLENDINDICES 格式 " + indices[1] + " 不支持"
    if weights is None or weights[1] not in SKIN_WEIGHT_FORMATS:
        return None, "BLENDWEIGHTS 格式 " + (weights[1] if weights else "缺失") + " 不支持"
    return (offset, weights[0], SKIN_WEIGHT_FORMATS[weights[1]], indices[0], False), None


def same_frame_skin_params(vertex_count: int, layout) -> list[int]:
    stride, weight_offset, weight_format, index_offset, one_bone = layout
    return [
        int(vertex_count),
        int(stride),
        int(weight_offset) | (int(weight_format) << 16),
        int(index_offset) | (0x10000 if one_bone else 0),
    ]


def reference_same_frame_skin(palette, position_bytes: bytes, blend_bytes: bytes, params,
                              found: bool = True) -> numpy.ndarray:
    """What the skinning shader writes: float32 rows of 10 values (zeros when bones are missing)."""
    vertex_count, stride, weight_word, index_word = (int(value) for value in params)
    if not found:
        return numpy.zeros((vertex_count, 10), dtype=numpy.float32)
    palette = numpy.asarray(palette, dtype=numpy.float64).reshape(-1, 4)
    vertices = numpy.frombuffer(position_bytes, dtype=numpy.float32)[:vertex_count * 10].reshape(-1, 10)
    rows = numpy.frombuffer(blend_bytes, dtype=numpy.uint8)[:vertex_count * stride].reshape(-1, stride)

    def bone_rows(bone):
        rows3 = numpy.zeros((3, 4))
        for row in range(3):
            slot = bone * 3 + row
            if 0 <= slot < len(palette):
                rows3[row] = palette[slot]
        return rows3

    index_offset = index_word & 0xFFFF
    weight_offset, weight_format = weight_word & 0xFFFF, weight_word >> 16
    skinned = numpy.zeros((vertex_count, 10), dtype=numpy.float32)
    for vertex in range(vertex_count):
        row = rows[vertex]
        if index_word & 0x10000:
            matrix = bone_rows(int(row[index_offset:index_offset + 4].view(numpy.uint32)[0]))
        else:
            bones = row[index_offset:index_offset + 16].view(numpy.uint32)
            raw = row[weight_offset:]
            if weight_format == 1:
                weights = raw[:16].view(numpy.float32).astype(numpy.float64)
            elif weight_format == 2:
                weights = raw[:8].view(numpy.float16).astype(numpy.float64)
            elif weight_format == 3:
                weights = raw[:8].view(numpy.uint16) / 65535.0
            else:
                weights = raw[:4].astype(numpy.float64) / 255.0
            matrix = sum(weights[k] * bone_rows(int(bones[k])) for k in range(4))
        position, normal, tangent = vertices[vertex, 0:3], vertices[vertex, 3:6], vertices[vertex, 6:10]
        skinned[vertex, 0:3] = matrix @ numpy.r_[position, 1.0]
        skinned[vertex, 3:6] = matrix[:, :3] @ normal
        skinned[vertex, 6:9] = matrix[:, :3] @ tangent[:3]
        skinned[vertex, 9] = tangent[3]
    return skinned


def instance_slot_count(participant_count: int) -> int:
    """Store slots: two stream-output buffers per copy of every participating component, twice over.

    Slots are given out first in, first out; the headroom keeps copies that come and go from
    reaching the slots of copies still on screen soon (unused slots expire).
    """
    return 4 * MAX_INSTANCES * max(1, int(participant_count))


def instance_slot_rows(profile: dict, component_ids) -> int:
    """Rows of a store slot: a header, then three per bone of the largest participant."""
    by_id = {component["component_id"]: component for component in profile["components"]}
    return 1 + 3 * max(int(by_id[int(component_id)]["vg_count"]) for component_id in component_ids)


def instance_params(profile: dict, component_id: int, slot_count: int, slot_rows: int) -> list[int]:
    """``uint4`` (table index, merged component count, slot count, rows per slot)."""
    return [runtime_component_index(profile, component_id), len(merged_components(profile)),
            int(slot_count), int(slot_rows)]


def instance_needed(profile: dict, component_ids) -> list[int]:
    """``uint4`` per merged component: (1 when a drawn part reads its bones, 0, 0, 0)."""
    wanted = {int(component_id) for component_id in component_ids}
    values = []
    for component in merged_components(profile):
        values += [1 if component["component_id"] in wanted else 0, 0, 0, 0]
    return values


def _rows_to_world(rows) -> numpy.ndarray:
    """4x4 column-convention matrix of object constants rows 0-3 (row vectors)."""
    rows = numpy.asarray(rows, dtype=numpy.float64).reshape(4, 4)
    matrix = numpy.eye(4)
    matrix[:3, :] = rows[:, :3].T
    return matrix


def reference_instance_match(profile: dict, store, records, params, needed, host_rows, host_slot: int,
                             frame: float) -> numpy.ndarray:
    """What the match shader writes: ``uint`` rows (1 + merged count, 4)."""
    host, count, slots, rows_per_slot = (int(value) for value in params)
    count = min(count, MAX_INSTANCE_COMPONENTS)
    store = numpy.asarray(store, dtype=numpy.float64).reshape(-1, 4)
    records = numpy.asarray(records, dtype=numpy.float64).reshape(-1, 4)
    needed = numpy.asarray(needed).reshape(-1, 4)[:, 0]
    table = numpy.asarray(runtime_component_table(profile), dtype=numpy.int64).reshape(-1, 4)

    def bone(slot, local):
        base = slot * rows_per_slot + 1 + local * 3
        return _to_affine44(store[base:base + 3])

    slot_of = [None] * count
    world = [numpy.eye(4)] * count
    host_found = (host < count and host_slot < slots
                  and store[host_slot * rows_per_slot, 0] == host and store[host_slot * rows_per_slot, 1] == frame)
    copies = sum(1 for h in range(slots)
                 if store[h * rows_per_slot, 0] == host and store[h * rows_per_slot, 1] == frame)
    if host_found:
        slot_of[host] = host_slot
        world[host] = _rows_to_world(host_rows)
    for _ in range(count if host_found else 0):
        progress = False
        for c in range(count):
            p = int(table[c * 2, 2])
            if p == c or p >= count:
                continue
            anchor = table[c * 2 + 1]
            if slot_of[p] is not None and slot_of[c] is None:
                known, target, known_anchor, target_anchor = p, c, int(anchor[0]), int(anchor[1])
            elif slot_of[c] is not None and slot_of[p] is None:
                known, target, known_anchor, target_anchor = c, p, int(anchor[1]), int(anchor[0])
            else:
                continue
            if needed[target] == 0:
                continue
            target_local = target_anchor - int(table[target * 2, 0])
            known_bone = bone(slot_of[known], known_anchor - int(table[known * 2, 0]))
            expected = (world[known] @ known_bone)[:3, 3]
            best, only, best_distance, second_distance = None, None, 3.0e38, 3.0e38
            candidates = unrecorded = 0
            for s in range(slots):
                header = store[s * rows_per_slot]
                if header[0] != target or header[1] != frame:
                    continue
                candidates += 1
                only = s
                record = records[s * INSTANCE_RECORD_ROWS + 4]
                if record[0] == target and record[2] == 1 and frame - record[1] <= 2:
                    recorded = _rows_to_world(records[s * INSTANCE_RECORD_ROWS:s * INSTANCE_RECORD_ROWS + 4])
                    distance = float(numpy.linalg.norm((recorded @ bone(s, target_local))[:3, 3] - expected))
                    if distance < best_distance:
                        best, best_distance, second_distance = s, distance, best_distance
                    elif distance < second_distance:
                        second_distance = distance
                else:
                    unrecorded += 1
            if candidates == 1 and copies <= 1:
                chosen = only
            elif best is not None and best_distance <= MATCH_DISTANCE:
                chosen = best
            elif best is not None and unrecorded == 0 and candidates > 1 and best_distance * 2.0 < second_distance:
                chosen = best
            else:
                continue
            slot_of[target] = chosen
            world[target] = world[known] @ known_bone @ numpy.linalg.inv(bone(chosen, target_local))
            progress = True
        if not progress:
            break
    choice = numpy.zeros((1 + count, 4), dtype=numpy.uint32)
    all_found = host_found
    for m in range(count):
        found = slot_of[m] is not None
        if needed[m] and not found:
            all_found = False
        choice[1 + m] = [slot_of[m] if found else 0, 1 if found else 0, 0, 0]
    choice[0, 0] = 1 if all_found else 0
    return choice


def reference_instance_gather(profile: dict, store, params, choice) -> numpy.ndarray:
    """What the gather shader writes: the pool of one copy, (bones_count, 3, 4)."""
    _, count, _, rows_per_slot = (int(value) for value in params)
    store = numpy.asarray(store, dtype=numpy.float64).reshape(-1, 4)
    pool = numpy.zeros((int(profile["bones_count"]), 3, 4))
    for k, component in enumerate(merged_components(profile)[:count]):
        slot, found = int(choice[1 + k][0]), int(choice[1 + k][1])
        if not found:
            continue
        for local in range(int(component["vg_count"])):
            base = slot * rows_per_slot + 1 + local * 3
            pool[component["vg_offset"] + local] = store[base:base + 3]
    return pool


def _is_placement(m) -> bool:
    det = numpy.linalg.det(m[:3, :3])
    return bool(numpy.all(numpy.abs(m[:3, 3]) < 1e-3) and abs(m[3, 3] - 1.0) < 1e-3
                and 0.25 < det < 4.0 and numpy.all(numpy.abs(m[3, :3]) < 1e6))


def reference_placement(self_rows, history, place: bool = True, shift=None) -> tuple:
    """What the placement shader writes: (29 object constant rows, 8 record rows or None, shift).

    ``history``: this part's records of the previous frame (rows 0-3 its matrix,
    4-7 the matrix it was drawn with).  A record of None leaves it unchanged.
    ``shift``: the distance of this frame set by a part drawn earlier, or None;
    the shift after this part is returned.
    """
    self_rows = numpy.asarray(self_rows, dtype=numpy.float64)
    history = numpy.asarray(history, dtype=numpy.float64)
    output = self_rows.copy()
    self_now = self_rows[0:4]
    if not _is_placement(self_now):
        return output, None, shift
    self_previous, placed_previous = history[0:4], history[4:8]
    placed = self_now
    if place and _is_placement(self_previous):
        if shift is None:
            shift = self_now[3, :3] - self_previous[3, :3]
        placed = self_previous.copy()
        placed[3, :3] += shift
        if not _is_placement(placed_previous):
            placed_previous = placed
        output[0:4] = placed
        output[4:8] = numpy.linalg.inv(placed)
        output[20:24] = placed_previous
    return output, numpy.concatenate([self_now, placed]), shift


def write_runtime_shaders(shader_folder: str, frame_delay: bool = False, same_frame: bool = False) -> list[str]:
    os.makedirs(shader_folder, exist_ok=True)
    paths = []
    shaders = [
        (IMPORT_SHADER_FILENAME, IMPORT_SHADER_SOURCE),
        (PALETTE_SHADER_FILENAME, PALETTE_SHADER_SOURCE),
    ]
    if frame_delay:
        shaders.append((DELAY_SHADER_FILENAME, DELAY_SHADER_SOURCE))
        shaders.append((PLACE_SHADER_FILENAME, PLACE_SHADER_SOURCE))
    if same_frame:
        shaders.append((SKIN_SHADER_FILENAME, SKIN_SHADER_SOURCE))
        shaders.append((INSTANCE_IMPORT_SHADER_FILENAME, INSTANCE_IMPORT_SHADER_SOURCE))
        shaders.append((INSTANCE_RECORD_SHADER_FILENAME, INSTANCE_RECORD_SHADER_SOURCE))
        shaders.append((INSTANCE_MATCH_SHADER_FILENAME, INSTANCE_MATCH_SHADER_SOURCE))
        shaders.append((INSTANCE_GATHER_SHADER_FILENAME, INSTANCE_GATHER_SHADER_SOURCE))
    for filename, source in shaders:
        path = os.path.join(shader_folder, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(source)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Reference implementation of the runtime (tests and extraction self-check)
# ---------------------------------------------------------------------------

def _to_affine44(rows: numpy.ndarray) -> numpy.ndarray:
    matrix = numpy.zeros(rows.shape[:-2] + (4, 4), dtype=numpy.float64)
    matrix[..., :3, :] = rows
    matrix[..., 3, 3] = 1.0
    return matrix


def reference_palette(profile: dict, pool: numpy.ndarray, component_id: int) -> numpy.ndarray:
    """What the palette shader writes for ``component_id`` (float64).

    ``pool`` holds ``bones_count`` bones of shape (3, 4).
    """
    merged = merged_components(profile)
    table = runtime_component_table(profile)
    pool = numpy.asarray(pool, dtype=numpy.float64).reshape(-1, 3, 4)

    def to_root(index: int) -> numpy.ndarray:
        accumulated = numpy.eye(4)
        for _ in range(32):
            parent = table[index * 8 + 2]
            if parent == index:
                break
            parent_slot, self_slot = table[index * 8 + 4], table[index * 8 + 5]
            edge = _to_affine44(pool[parent_slot]) @ numpy.linalg.inv(_to_affine44(pool[self_slot]))
            accumulated = edge @ accumulated
            index = parent
        return accumulated

    own_index = runtime_component_index(profile, component_id)
    to_self_root_inverse = numpy.linalg.inv(to_root(own_index))
    palette = pool.copy()
    for index, component in enumerate(merged):
        if index == own_index:
            continue
        conversion = to_self_root_inverse @ to_root(index)
        rows = slice(component["vg_offset"], component["vg_offset"] + component["vg_count"])
        palette[rows] = (conversion[None] @ _to_affine44(pool[rows]))[:, :3, :]
    return palette


# ---------------------------------------------------------------------------
# Extraction-time planning
# ---------------------------------------------------------------------------

def find_world_matches(world_a: numpy.ndarray, world_b: numpy.ndarray) -> tuple[list, int]:
    """Bones of a and b whose world space skinning matrices agree.

    Returns ``(mutual_unique_pairs, all_pair_count)``.  A pair is mutual-unique
    when neither bone matches any other bone of the other component; bones that
    are identical only because they do not move in the captured frame match
    several bones at once and are not used.
    """
    rows_a = world_a[:, :3, :]
    rows_b = world_b[:, :3, :]
    difference = numpy.abs(rows_a[:, None] - rows_b[None])
    rotation_error = difference[..., :3].max(axis=(2, 3))
    translation_error = difference[..., 3].max(axis=2)
    magnitude = numpy.maximum(
        numpy.abs(rows_a[:, :, 3]).max(axis=1)[:, None],
        numpy.abs(rows_b[:, :, 3]).max(axis=1)[None],
    )
    tolerance = MATCH_TRANSLATION_TOLERANCE + MATCH_TRANSLATION_RELATIVE_TOLERANCE * magnitude
    nonzero_a = numpy.abs(rows_a).max(axis=(1, 2)) > 0
    nonzero_b = numpy.abs(rows_b).max(axis=(1, 2)) > 0
    matches = (
        (rotation_error < MATCH_ROTATION_TOLERANCE)
        & (translation_error < tolerance)
        & nonzero_a[:, None] & nonzero_b[None]
    )
    row_counts = matches.sum(axis=1)
    column_counts = matches.sum(axis=0)
    unique_pairs = [
        (int(i), int(j)) for i, j in numpy.argwhere(matches)
        if row_counts[i] == 1 and column_counts[j] == 1
    ]
    return unique_pairs, int(matches.sum())


def plan_merged_skeleton(candidates: list[dict]) -> tuple[dict, list[str]]:
    """Choose the character, the merged components and the shared skeleton.

    Each candidate is a dict with:
        ib_hash, unique_strs, blend_vb, pose_vs, layout_signature, pose_order, vertex_count,
        indices_per_vertex, palette (n, 3, 4) float32,
        object_matrix (4, 4) float64 or None (column vectors),
        shape_keyed (bool), weighted_counts (n,) int.

    Returns ``(profile_without_validation, warnings)``.  Components of the
    returned profile are ordered: merged ones first (root first, then in
    capture order), then the others.
    """
    warnings: list[str] = []
    usable = [
        candidate for candidate in candidates
        if candidate.get("object_matrix") is not None and len(candidate["palette"])
    ]
    for candidate in candidates:
        if candidate not in usable:
            warnings.append(
                candidate["ib_hash"] + " 读不到骨骼表或物体矩阵，不参与骨骼合并。"
            )
    if not usable:
        raise ZZMIMergedSkeletonError("帧分析里没有可用于骨骼合并的角色部件。")

    world = {}
    for candidate in usable:
        palette = numpy.asarray(candidate["palette"], dtype=numpy.float64).reshape(-1, 3, 4)
        world[candidate["ib_hash"]] = numpy.einsum(
            "ij,bjk->bik", numpy.asarray(candidate["object_matrix"], dtype=numpy.float64),
            _to_affine44(palette),
        )

    pair_info = {}
    for index_a, candidate_a in enumerate(usable):
        for candidate_b in usable[index_a + 1:]:
            unique_pairs, all_count = find_world_matches(
                world[candidate_a["ib_hash"]], world[candidate_b["ib_hash"]]
            )
            if all_count:
                key = (candidate_a["ib_hash"], candidate_b["ib_hash"])
                pair_info[key] = (unique_pairs, all_count)
                pair_info[(key[1], key[0])] = ([(j, i) for i, j in unique_pairs], all_count)

    rigged = [candidate for candidate in usable if not candidate["shape_keyed"]]
    if not rigged:
        raise ZZMIMergedSkeletonError("帧分析里只有形态键驱动的部件，没有可合并的骨骼。")
    root = max(rigged, key=lambda candidate: (candidate["vertex_count"], -candidate["pose_order"]))

    # Grow the merged set from the root over mutual-unique anchors, always
    # taking the edge with the most shared bones (a maximum spanning tree).
    parent_of = {root["ib_hash"]: (root["ib_hash"], None)}
    by_ib = {candidate["ib_hash"]: candidate for candidate in usable}
    while True:
        best = None
        for (ib_a, ib_b), (unique_pairs, _) in pair_info.items():
            if ib_a not in parent_of or ib_b in parent_of or not unique_pairs:
                continue
            if by_ib[ib_b]["shape_keyed"]:
                continue
            rank = (len(unique_pairs), -by_ib[ib_b]["pose_order"])
            if best is None or rank > best[0]:
                best = (rank, ib_a, ib_b, unique_pairs)
        if best is None:
            break
        _, ib_parent, ib_child, unique_pairs = best
        weights_parent = by_ib[ib_parent]["weighted_counts"]
        weights_child = by_ib[ib_child]["weighted_counts"]
        # The anchor should be a bone both meshes really use (head, neck)
        # rather than a helper bone that happens to be unique.
        anchor = max(
            unique_pairs,
            key=lambda pair: (
                min(_count(weights_parent, pair[0]), _count(weights_child, pair[1])),
                -pair[0],
            ),
        )
        parent_of[ib_child] = (ib_parent, anchor)

    merged_ibs = set(parent_of)
    attached = []
    for candidate in usable:
        ib_hash = candidate["ib_hash"]
        if ib_hash in merged_ibs:
            continue
        touches_character = any(
            (ib_hash, merged_ib) in pair_info for merged_ib in merged_ibs
        )
        if not touches_character:
            warnings.append(
                ib_hash + " 和角色没有共用的骨骼，判断为画面里的其它物体 (邦布、武器或其它角色)，没有提取。"
            )
            continue
        if candidate["shape_keyed"]:
            role, reason = ROLE_SHAPE_KEYED, "形态键驱动 (脸部表情)"
        else:
            role, reason = ROLE_STANDALONE, "和合并部件没有一一对应的共用骨骼，无法换算骨骼空间"
        attached.append((candidate, role, reason))

    # Merged components: root first, then by capture order.
    merged_order = [root] + sorted(
        (by_ib[ib_hash] for ib_hash in merged_ibs if ib_hash != root["ib_hash"]),
        key=lambda candidate: candidate["pose_order"],
    )
    offsets = {}
    vg_offset = 0
    for candidate in merged_order:
        offsets[candidate["ib_hash"]] = vg_offset
        vg_offset += len(candidate["palette"])
    bones_count = vg_offset
    if bones_count > MAX_POOL_BONES:
        raise ZZMIMergedSkeletonError(
            "合并后的骨骼数 " + str(bones_count) + " 超过游戏蒙皮着色器上限 " + str(MAX_POOL_BONES) + "。"
        )

    # Canonical ids: union of mutual-unique matches between merged components.
    # The representative is the copy uploaded earliest in the frame.
    union_parent = list(range(bones_count))

    def find(slot):
        while union_parent[slot] != slot:
            union_parent[slot] = union_parent[union_parent[slot]]
            slot = union_parent[slot]
        return slot

    for (ib_a, ib_b), (unique_pairs, _) in pair_info.items():
        if ib_a not in merged_ibs or ib_b not in merged_ibs or ib_a > ib_b:
            continue
        for i, j in unique_pairs:
            root_a, root_b = find(offsets[ib_a] + i), find(offsets[ib_b] + j)
            if root_a != root_b:
                union_parent[max(root_a, root_b)] = min(root_a, root_b)

    order_of_slot = {}
    for candidate in merged_order:
        for local_id in range(len(candidate["palette"])):
            order_of_slot[offsets[candidate["ib_hash"]] + local_id] = (
                candidate["pose_order"], -_count(candidate["weighted_counts"], local_id),
                offsets[candidate["ib_hash"]] + local_id,
            )
    members = {}
    for slot in range(bones_count):
        members.setdefault(find(slot), []).append(slot)
    canonical_of = {}
    for slot_group in members.values():
        representative = min(slot_group, key=lambda slot: order_of_slot[slot])
        for slot in slot_group:
            canonical_of[slot] = representative

    components = []
    for candidate in merged_order:
        ib_hash = candidate["ib_hash"]
        parent_ib, anchor = parent_of[ib_hash]
        offset = offsets[ib_hash]
        components.append({
            "ib_hash": ib_hash,
            "role": ROLE_MERGED,
            "reason": "",
            "vg_offset": offset,
            "vg_count": len(candidate["palette"]),
            "vg_map": {
                str(local_id): int(canonical_of[offset + local_id])
                for local_id in range(len(candidate["palette"]))
            },
            "_parent_ib": parent_ib,
            "anchor_parent_slot": (offsets[parent_ib] + anchor[0]) if anchor else 0,
            "anchor_self_slot": (offset + anchor[1]) if anchor else 0,
            **_common_fields(candidate),
        })
    for candidate, role, reason in sorted(attached, key=lambda item: item[0]["pose_order"]):
        components.append({
            "ib_hash": candidate["ib_hash"], "role": role, "reason": reason,
            **_common_fields(candidate),
        })

    id_by_ib = {component["ib_hash"]: index for index, component in enumerate(components)}
    for index, component in enumerate(components):
        component["component_id"] = index
        parent_ib = component.pop("_parent_ib", None)
        component["parent_component_id"] = id_by_ib[parent_ib] if parent_ib else NO_PARENT

    for component in components:
        if component["role"] == ROLE_SHAPE_KEYED:
            warnings.append(
                component["ib_hash"] + " 是形态键驱动的部件 (脸部表情)，不参与骨骼合并；"
                "导出时保持游戏原网格，只替换贴图。"
            )
        elif component["role"] == ROLE_STANDALONE:
            warnings.append(
                component["ib_hash"] + " " + component["reason"] + "，不参与骨骼合并，按普通方式导出。"
            )

    profile = {
        "format_version": PROFILE_FORMAT_VERSION,
        "mode": PROFILE_MODE,
        "components": components,
    }
    return profile, warnings


def _count(weighted_counts, local_id: int) -> int:
    if weighted_counts is None or local_id >= len(weighted_counts):
        return 0
    return int(weighted_counts[local_id])


def _common_fields(candidate: dict) -> dict:
    return {
        "unique_strs": list(candidate["unique_strs"]),
        "blend_vb": candidate.get("blend_vb", ""),
        "pose_vs": candidate.get("pose_vs", ""),
        "layout_signature": candidate.get("layout_signature", ""),
        "pose_order": int(candidate.get("pose_order", 0)),
        "vertex_count": int(candidate.get("vertex_count", 0)),
        "indices_per_vertex": int(candidate.get("indices_per_vertex", 4)),
        "palette_bone_count": int(len(candidate["palette"])),
        "pose_vertex_count": int(candidate.get("pose_vertex_count", 0)),
    }
