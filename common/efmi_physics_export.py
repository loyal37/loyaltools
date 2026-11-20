"""Optional EFMI physics export preparation, independent of Blender and disk I/O.

Authoring vertex-group names and IDs are stable. The runtime palette is dense;
neither Blender's group indices nor a component's LoD table defines those IDs.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import copy
import json
import os
import re
import tempfile

import numpy as np

PHYSICS_GROUP_PREFIX = "EFMI_PHYS_"
MAX_AUTHORING_ID = 2147483647


def _is_physics_group(name):
    return bool(re.fullmatch(r"EFMI_PHYS_[0-9]+", str(name)))


def _require(condition, message):
    if not condition:
        raise ValueError("EFMI 物理导出: " + message)


def _matrix(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), label + " 必须是有限 4×4 矩阵。")
    _require(np.allclose(matrix[3], [0, 0, 0, 1]), label + " 必须是仿射矩阵。")
    _require(abs(np.linalg.det(matrix[:3, :3])) > 1e-10, label + " 不能包含零缩放。")
    return matrix


def _point(matrix, value, label):
    value = np.asarray(value, dtype=np.float64)
    _require(value.shape == (3,) and np.isfinite(value).all(), label + " 必须是有限三维坐标。")
    return (matrix @ np.append(value, 1.0))[:3]


def prepare_physics_plan(records, profile, *, mirror_x=False):
    """Validate all records before export writes, then convert rest frames to model space."""
    if not records or not records.get("enabled") or not records.get("bones"):
        return None
    _require(profile is not None, "只支持 EFMI 骨骼合并导出。")
    base = int(profile["bones_count"])
    _require(int(records.get("base_bone_count", -1)) == base,
             "原骨数量 B 与当前工作空间不一致，请重新同步物理骨。")
    result = copy.deepcopy(records)
    export_transform = np.diag([-1.0, 1.0, 1.0, 1.0]) if mirror_x else np.eye(4)
    result["export_space_transform"] = export_transform.tolist()
    bones = result["bones"]
    ids = [int(bone["global_id"]) for bone in bones]
    _require(len(ids) == len(set(ids)), "物理骨全局编号重复。")
    _require(all(base <= bone_id <= MAX_AUTHORING_ID for bone_id in ids),
             "物理骨内部编号必须在原骨范围 B 之外，且不超过 2^31-1。")
    _require(base + len(bones) <= 65536, "扩展骨骼总数超过 R16_UINT 容量。")
    id_to_palette = {bone_id: base + offset for offset, bone_id in enumerate(sorted(ids))}
    by_id = {int(bone["global_id"]): bone for bone in bones}
    group_names = {int(bone["global_id"]): str(bone.get("group_name", bone["global_id"])) for bone in bones}
    _require(len(set(group_names.values())) == len(group_names), "物理骨顶点组名称重复。")
    for bone_id, name in group_names.items():
        if name.isdecimal():
            _require(int(name) == bone_id and bone_id <= 65535, "旧数字物理顶点组与骨编号不一致。")
        else:
            _require(_is_physics_group(name) and bone_id >= 65536, "物理组必须使用专属前缀及独立内部编号。")
    chains = result.get("chains", [])
    _require(bool(chains), "物理骨缺少所属链。")
    chain_ids = [chain["chain_id"] for chain in chains]
    _require(len(set(chain_ids)) == len(chain_ids), "物理链标识重复。")
    claimed = set()
    gpu_components = [component for component in profile["components"] if not component["cpu_posed"]]
    for chain_index, chain in enumerate(chains):
        linked = bool(chain.get("animation_anchor_group"))
        native = int(chain.get("follow_group_id", chain.get("anchor", -1)))
        if linked and native != -1:
            raise ValueError("EFMI physics export: A physics chain can use only one attachment.")
        # The animation bone already follows a native palette source. This
        # index tracks freshness only and does not become an authored mount.
        anchor = int(chain["animation_source_anchor"]) if linked else native
        _require(0 <= anchor < base, "链 " + str(chain["chain_id"]) + " 的挂点不在原骨范围。")
        owners = [component for component in gpu_components
                  if int(component["vg_offset"]) <= anchor < int(component["vg_offset"]) + int(component["vg_count"])]
        _require(len(owners) == 1, "链挂点没有唯一 GPU 原骨组件。")
        segment_ids = [int(item.get("global_id", item.get("bone_id"))) if isinstance(item, dict) else int(item)
                       for item in chain.get("bone_ids", chain.get("segments", []))]
        _require(bool(segment_ids), "物理链必须至少包含一个真实骨骼。")
        first_parent = -1 if linked else anchor
        previous = first_parent
        for bone_id in segment_ids:
            _require(bone_id in by_id and bone_id not in claimed, "链引用未知骨骼或重复占用骨骼 " + str(bone_id) + "。")
            bone = by_id[bone_id]
            _require(bone.get("chain_id") == chain["chain_id"], "骨骼与所属物理链不一致。")
            parent = bone.get("parent_global_id", previous)
            parent = first_parent if parent is None and previous == first_parent else parent
            _require(parent is not None and int(parent) == previous, "骨骼父子顺序无效，请同步物理链。")
            bone["parent_global_id"] = int(parent)
            bone["parent_index"] = -1 if previous == first_parent else id_to_palette[previous] - base
            bone["palette_index"] = id_to_palette[bone_id]
            bone["chain_index"] = chain_index
            rig_matrix = export_transform @ _matrix(bone.get("rig_matrix_world", np.eye(4)), "骨架变换")
            local_rest = _matrix(bone["rest_matrix"], "骨骼静止变换")
            model_rest = rig_matrix @ local_rest
            head = _point(rig_matrix, bone["head"], "骨骼头部")
            tail = _point(rig_matrix, bone["tail"], "骨骼尾部")
            length = float(np.linalg.norm(tail - head))
            _require(length > 1e-7, "物理骨 " + str(bone_id) + " 长度为零。")
            # The solver constructs rigid pose frames.  Preserve the edited
            # direction and roll while baking rig scale into segment length.
            y_axis = (tail - head) / length
            x_axis = model_rest[:3, 0] - y_axis * np.dot(model_rest[:3, 0], y_axis)
            _require(np.linalg.norm(x_axis) > 1e-7, "物理骨静止坐标轴退化。")
            x_axis /= np.linalg.norm(x_axis)
            model_rest[:3, :3] = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
            model_rest[:3, 3] = head
            bone["rest_matrix_local"] = local_rest.tolist()
            bone["rest_matrix"] = model_rest.tolist()
            bone["inverse_bind"] = np.linalg.inv(model_rest).tolist()
            bone["head"] = head.tolist()
            bone["tail"] = tail.tolist()
            bone["length"] = length
            bone.setdefault("params", chain.get("params", {}))
            bone["root_head"] = (_point(rig_matrix, bone["root_head"], "链根部").tolist()
                                 if "root_head" in bone else head.tolist())
            claimed.add(bone_id)
            previous = bone_id
        chain.update(anchor=anchor, component_id=int(owners[0]["component_id"]),
                     segments=segment_ids, root_head=by_id[segment_ids[0]]["head"])
    _require(claimed == set(ids), "存在未绑定到物理链的额外骨骼。")
    bones.sort(key=lambda bone: bone["palette_index"])
    meshes = result.get("meshes", [])
    _require(bool(meshes), "物理骨没有绑定导出网格。")
    group_to_palette = {}
    for mesh in meshes:
        mesh["object_matrix_world"] = _matrix(mesh["object_matrix_world"], "网格变换").tolist()
        active_names = set(mesh.get("active_group_names", mesh.get("vg_mapping", {})))
        mesh["palette_mapping"] = {}
        for name, bone_id in mesh.get("vg_mapping", {}).items():
            name, bone_id = str(name), int(bone_id)
            native = name.isdecimal() and int(name) == bone_id and 0 <= bone_id < base
            custom = bone_id in by_id and group_names[bone_id] == name
            if not native and not custom:
                _require(name not in active_names, "顶点组 " + repr(name) + " 未绑定有效的原生骨或专属物理骨。")
                continue
            palette_id = id_to_palette.get(bone_id, bone_id)
            _require(name not in group_to_palette or group_to_palette[name] == palette_id,
                     "多个网格使用了同名但不同骨骼的物理顶点组 " + name + "。")
            group_to_palette[name] = palette_id
            mesh["palette_mapping"][name] = palette_id
    result.update(base_bone_count=base, bone_count=len(bones), extended_bone_count=base + len(bones),
                  global_to_palette={str(key): value for key, value in id_to_palette.items()})
    result["group_to_palette"] = group_to_palette
    return result


def resolve_plan_anchors(plan, profile, object_components, lod_bone_errors, resolver):
    """Change only the export plan; preserve authoring IDs, groups and rest pose.

    A chain can be consumed by multiple meshes and multiple blueprint draws.
    All of those draws constrain its one stable runtime provider.
    """
    by_id = {int(bone["global_id"]): bone for bone in plan["bones"]}
    for chain in plan["chains"]:
        if chain.get("animation_anchor_group"):
            continue
        segment_ids = set(chain["segments"])
        consumers = []
        for mesh in plan["meshes"]:
            active = set(mesh.get("active_group_names", mesh.get("vg_mapping", {})))
            if not any(name in active and int(value) in segment_ids
                       for name, value in mesh.get("vg_mapping", {}).items()):
                continue
            names = [mesh["object_name"], *mesh.get("export_object_names", [])]
            for name in names:
                consumers.extend(object_components.get(name, [None]))
        original = int(chain["anchor"])
        anchor, component_id, unknown = resolver(profile, original, consumers, lod_bone_errors)
        chain["authoring_anchor"] = original
        chain["anchor"] = anchor
        chain["component_id"] = component_id
        by_id[chain["segments"][0]]["parent_global_id"] = anchor
        if anchor != original:
            print(f"[EFMI物理] 链 {chain['chain_id']}: 挂点 {original} → {anchor}，更新来源组件 {component_id}。")
        if unknown:
            print(f"[EFMI物理][警告] 链 {chain['chain_id']}: 挂点 {anchor} 缺少 LOD {unknown} 的骨骼核对数据，请补全 LOD 预览后验证。")


def validate_object_weights(objects, plan, *, animation_weights=None):
    """Reject unknown weighted groups, including unmanaged copies carrying custom IDs."""
    base = plan["base_bone_count"]
    registered = {name for mesh in plan["meshes"] for name in (mesh["object_name"], *mesh.get("export_object_names", []))}
    custom_ids = {int(key) for key in plan["global_to_palette"]}
    mesh_mapping = {mesh["object_name"]: mesh.get("vg_mapping", {}) for mesh in plan["meshes"]}
    for obj in objects:
        groups = {int(group.index): str(group.name) for group in obj.vertex_groups}
        numeric_ids = [int(name) for name in groups.values() if name.isdecimal()]
        _require(len(numeric_ids) == len(set(numeric_ids)), obj.name + " 存在多个同编号顶点组。")
        for vertex in obj.data.vertices:
            for weight in vertex.groups:
                if float(weight.weight) <= 0:
                    continue
                name = groups.get(int(weight.group), "")
                if animation_weights and name in animation_weights.get(obj.name, ()):
                    continue
                explicit = mesh_mapping.get(obj.name, {})
                _require(name.isdecimal() or name in explicit,
                         obj.name + " 的顶点组 " + repr(name) + " 含有未绑定的权重。")
                bone_id = int(explicit[name]) if name in explicit else int(name)
                if bone_id >= base:
                    _require(bone_id in custom_ids, obj.name + " 引用了未知物理骨 " + name + "。")
                    _require(obj.name in registered, obj.name + " 未绑定物理骨架，请先登记或移除额外权重。")


def verified_physics_binding(obj, settings, *, is_export_copy=None, allow_inherited=False):
    """Read-only ownership proof; a prefix or copied custom property is insufficient."""
    for entry in getattr(settings, "objects", ()):
        source, rig = entry.object, entry.rig
        if source is None or rig is None:
            continue
        if obj != source and not (is_export_copy and is_export_copy(entry, obj)) and not allow_inherited:
            continue
        if source.get("LoyalTools:EFMIPhysicsObject") != entry.uid or obj.get("LoyalTools:EFMIPhysicsObject") != entry.uid:
            continue
        if not rig.get("LoyalTools:EFMIPhysicsRig") or rig.get("LoyalTools:EFMIPhysicsOwner") != entry.uid:
            continue
        segments = [segment for chain in entry.chains for segment in chain.segments]
        numbers = [int(segment.bone_id) for segment in segments]
        names = {segment.group_name or str(segment.bone_id) for segment in segments}
        try:
            stored = {int(value) for value in json.loads(source.get("LoyalTools:EFMIPhysicsOwnedGroups", "[]"))}
            target_stored = {int(value) for value in json.loads(obj.get("LoyalTools:EFMIPhysicsOwnedGroups", "[]"))}
            if len(numbers) != len(set(numbers)) or set(numbers) != stored or stored != target_stored:
                continue
            if getattr(entry, "schema_version", 0) >= 2:
                if set(json.loads(entry.owned_group_ids)) != stored or names & set(json.loads(entry.native_group_names)):
                    continue
            recorded_names = source.get("LoyalTools:EFMIPhysicsOwnedGroupNames")
            if recorded_names is not None and set(json.loads(recorded_names)) != names:
                continue
        except (TypeError, ValueError):
            continue
        legacy_segments = [(chain, segment) for chain in entry.chains for segment in chain.segments
                           if (segment.group_name or str(segment.bone_id)).isdecimal()]
        if legacy_segments:
            bones = getattr(getattr(rig, "data", None), "bones", ())
            if any(not any(bone.get("LoyalTools:EFMIPhysicsSegment") == segment.uid
                           and bone.get("LoyalTools:EFMIPhysicsChain") == chain.uid for bone in bones)
                   for chain, segment in legacy_segments):
                # Legacy numeric metadata alone cannot claim a native group.
                continue
        return names, source
    return None


def verified_physics_group_names(obj, settings, **kwargs):
    binding = verified_physics_binding(obj, settings, **kwargs)
    return binding[0] if binding else set()


def reject_disabled_physics_weights(objects, *, owned_group_names=None, owned_sources=None, merged_mode=True):
    """Read-only safety check: OFF must not silently export custom IDs as native."""
    for obj in objects:
        owned_names = (owned_group_names or {}).get(obj.name, set())
        if not owned_names:
            continue
        source = (owned_sources or {}).get(obj.name)
        for target in (obj,) if source is None or source == obj else (obj, source):
            owned_indices = {int(group.index) for group in target.vertex_groups if group.name in owned_names}
            if not owned_indices:
                continue
            for vertex in target.data.vertices:
                if any(int(weight.group) in owned_indices and float(weight.weight) > 0 for weight in vertex.groups):
                    raise ValueError(
                        "EFMI 物理导出: " + obj.name + " 仍有物理顶点组权重。"
                        + ("请开启“启用 EFMI 物理模块”并登记物体，" if merged_mode else "请使用 EFMI 骨骼合并模式，") +
                        "或先将物理权重恢复到原生顶点组。本次导出已取消，原导出文件未修改。"
                    )


def remap_joined_vertex_groups(category_buffers, game_type, vertex_groups, plan, *, active_group_indices=None):
    """Rewrite joined-object group indices using its actual names, never index==name."""
    if not plan:
        return
    index_to_id = {}
    for group in vertex_groups:
        name = str(group.name)
        if name in plan.get("group_to_palette", {}):
            index_to_id[int(group.index)] = int(plan["group_to_palette"][name])
        elif name.isdecimal():
            stable_id = int(name)
            _require(stable_id < plan["base_bone_count"] or (active_group_indices is not None and int(group.index) not in active_group_indices),
                     "合并网格包含未知额外顶点组 " + name + "。")
            index_to_id[int(group.index)] = stable_id if stable_id < plan["base_bone_count"] else 0
        elif _is_physics_group(name) or name.startswith("EFMI_ANIM_"):
            if active_group_indices is None or int(group.index) in active_group_indices:
                raise ValueError("EFMI export: Vertex group " + name + " has no registered bone in the merged mesh.")
            index_to_id[int(group.index)] = 0
    for element in game_type.D3D11ElementList:
        if element.SemanticName != "BLENDINDICES":
            continue
        _require(element.Format == "R16G16B16A16_UINT", "物理网格需要 R16G16B16A16_UINT 骨索引。")
        category = element.Category
        stride = int(game_type.CategoryStrideDict[category])
        offset = 0
        for previous in game_type.D3D11ElementList:
            if previous is element:
                break
            if previous.Category == category:
                offset += int(previous.ByteWidth)
        rows = np.ascontiguousarray(category_buffers[category], dtype=np.uint8).copy().reshape(-1, stride)
        values = np.ascontiguousarray(rows[:, offset:offset + int(element.ByteWidth)]).view(np.uint16)
        original = values.copy()
        for group_index in np.unique(original):
            # Padding index 0 is permitted on unweighted meshes without group 0.
            _require(int(group_index) in index_to_id or int(group_index) == 0,
                     "导出包含未映射顶点组索引 " + str(group_index) + "。")
            values[original == group_index] = index_to_id.get(int(group_index), 0)
        rows[:, offset:offset + int(element.ByteWidth)] = values.view(np.uint8).reshape(len(rows), -1)
        category_buffers[category] = rows.ravel()


def base_bone_mask(values, active_mask, plan):
    """Keep extended palette entries out of private-pool/LoD audits and rewrites."""
    if not plan:
        return active_mask
    mask = np.asarray(values) < int(plan["base_bone_count"])
    return mask if active_mask is None else np.asarray(active_mask, dtype=bool) & mask


def build_anchor_fallback_buffers(category_buffers, game_type, plan):
    """Make draw-local native-B-safe Blend buffers for unsupported runtime paths.

    Only custom index channels change. Weight bytes and every other semantic are
    copied verbatim, preserving ordinary geometry sharing the same submesh.
    """
    if not plan:
        return {}
    mapping = {}
    for chain in plan["chains"]:
        if chain.get("animation_anchor_group"):
            continue
        for global_id in chain["segments"]:
            mapping[int(plan["global_to_palette"][str(global_id)])] = int(chain["anchor"])
    result = {}
    for element in game_type.D3D11ElementList:
        if element.SemanticName != "BLENDINDICES":
            continue
        _require(element.Format == "R16G16B16A16_UINT", "物理降级缓冲需要 R16 骨索引。")
        category = element.Category
        stride = int(game_type.CategoryStrideDict[category])
        offset = 0
        for previous in game_type.D3D11ElementList:
            if previous is element:
                break
            if previous.Category == category:
                offset += int(previous.ByteWidth)
        rows = np.ascontiguousarray(result.get(category, category_buffers[category]), dtype=np.uint8).copy().reshape(-1, stride)
        width = int(element.ByteWidth)
        values = np.ascontiguousarray(rows[:, offset:offset + width]).view(np.uint16)
        original = values.copy()
        for custom_id, anchor in mapping.items():
            values[original == custom_id] = anchor
        if not np.array_equal(values, original):
            rows[:, offset:offset + width] = values.view(np.uint8).reshape(len(rows), width)
            result[category] = rows.ravel()
    return result


def runtime_profile(profile):
    # Existing profiles predate posing-source metadata. Their rolling contract
    # is checked at runtime; an explicitly recorded CB/unknown source is never
    # relabelled as rolling to make generation succeed.
    source = profile.get("bones_data_source", profile.get("input_contract", {}).get("validated_source", "rolling"))
    sources = [source] + [component["bones_data_source"] for component in profile["components"]
                          if not component["cpu_posed"] and "bones_data_source" in component]
    _require(all(not isinstance(value, bool) and str(value).casefold() in {"1", "rolling", "efmi-rolling-buffer-v1"}
                 for value in sources), "当前 profile 明确记录了 CB 或未知骨骼来源，只支持 rolling buffer。")
    return {
        "adapter": {"kind": "efmi-rolling-buffer-v1", "original_bones": int(profile["bones_count"]),
                    "instances": int(profile["max_instance_count"]), "implicit_bones": 512},
        "input_contract": {"validated_source": "rolling", "matrix": "affine-3-float4-rows",
                           "bone_index_format": "R16G16B16A16_UINT"},
        "components": [{"id": int(component["component_id"]), "offset": int(component["vg_offset"]),
                        "count": int(component["vg_count"]), "declared_gpu_posed": True,
                        "draw_command": "CommandList_Draw_Component" + str(component["component_id"])}
                       for component in profile["components"] if not component["cpu_posed"]],
        "source_palette": "ResourceMergedSkeletonDataRW",
        "source_update_pool": "Pool_MergedSkeleton_Instance_UpdateFrame",
        "source_lod_pool": "Pool_MergedSkeleton_Instance_LodLevel",
    }


def serialize_authoring_records(plan):
    """Retain edited rig/mesh transforms and explicit stable-ID mapping for review."""
    return json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def split_runtime_sections(ini_text):
    """Merge singleton sections into the host INI instead of defining duplicates."""
    constants, present, remaining = [], [], []
    destination = remaining
    for line in ini_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].casefold()
            destination = constants if name == "constants" else present if name == "present" else remaining
            if name in {"constants", "present"}:
                continue
        destination.append(line)
    return constants, present, remaining


def write_runtime_files(output_folder, files):
    """Replace runtime files only after every temporary output has been prepared."""
    output = Path(output_folder).resolve()
    staged = []
    try:
        for relative_name, content in files.items():
            target = (output / relative_name).resolve()
            _require(output in target.parents, "运行时输出路径超出模组目录。")
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".efmi-physics-", suffix=".tmp", dir=target.parent)
            staged.append((temporary, target))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content.encode("utf-8") if isinstance(content, str) else content)
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


@contextmanager
def atomic_ini_target(target):
    """The active INI is replaced only after rendering and runtime files succeed."""
    target = Path(target)
    descriptor, temporary = tempfile.mkstemp(prefix=".efmi-physics-", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
