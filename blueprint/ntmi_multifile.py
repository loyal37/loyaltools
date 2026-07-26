from __future__ import annotations

from collections import OrderedDict, defaultdict
import glob
import os
import shutil

import bpy
import numpy as np

from .direct_export_multifile import DirectMultiFileGenerator, MultiFileDirectExportError
from ..common.d3d11_gametype import D3D11GameType
from ..ui.ntmi_modimp.ini_swap_patcher import ACTIVE_FLAG
from .ntmi_layout_adapter import iter_name_variants, parse_ntmi_part_layouts
from .ntmi_shapekey import _build_exported_loop_indices, _load_ntmi_exporter_module, _load_ntmi_position_converter


def _position_format_from_stride(position_stride: int) -> str:
    if int(position_stride) == 16:
        return "R32G32B32A32_FLOAT"
    if int(position_stride) == 8:
        return "R16G16B16A16_FLOAT"
    return "R32G32B32_FLOAT"


def _build_minimal_position_game_type(position_stride: int) -> D3D11GameType:
    return D3D11GameType.from_submesh_json_dict(
        {
            "WorkGameType": "NTMI_DIRECT",
            "GPU-PreSkinning": False,
            "CategoryDrawCategoryMap": {"Position": "Position"},
        },
        override_d3d11_element_list=[
            {
                "SemanticName": "POSITION",
                "SemanticIndex": 0,
                "Format": _position_format_from_stride(position_stride),
                "ByteWidth": int(position_stride),
                "ExtractSlot": "vb0",
                "ExtractTechnique": "trianglelist",
                "Category": "Position",
            }
        ],
    )


def _normalized_active_variable(raw_value: str) -> str:
    active_variable = str(raw_value or "").strip() or ACTIVE_FLAG
    if not active_variable.startswith("$"):
        active_variable = f"${active_variable}"
    if active_variable == "$active0":
        return ACTIVE_FLAG
    return active_variable

class NTMIDirectMultiFileGenerator(DirectMultiFileGenerator):
    def __init__(self, config_node, multi_file_nodes, mod_export_path: str, exporter):
        super().__init__(config_node, multi_file_nodes, mod_export_path, exporter)
        ini_files = glob.glob(os.path.join(mod_export_path, "*.ini"))
        if not ini_files:
            raise MultiFileDirectExportError("NTMI MultiFile: no ini file was found in the output directory.")

        self.target_ini_file = ini_files[0]
        self.sections, self.preserved_tail_content = self.config_node._read_ini_to_ordered_dict(self.target_ini_file)
        self.part_layouts = parse_ntmi_part_layouts(
            self.sections,
            output_dir=mod_export_path,
            source_ini_path=self.target_ini_file,
        )
        self.meshes_dir = os.path.join(mod_export_path, "Buffer")
        self.mod_importer_root = str(getattr(exporter, "mod_importer_root", "") or "").strip()
        self._modimp_exporter_module = _load_ntmi_exporter_module(self.mod_importer_root)
        self._position_converter = _load_ntmi_position_converter(mod_export_path, self.mod_importer_root)
        self._object_to_part_token = {}
        for part_token, part_layout in self.part_layouts.items():
            for draw_call in part_layout.draw_calls:
                for candidate_name in iter_name_variants(draw_call.mesh_name):
                    self._object_to_part_token.setdefault(candidate_name, part_token)

    def _resolve_part_token(self, obj_name: str) -> str:
        clean_name = str(obj_name or "").strip()
        if not clean_name:
            return ""
        if clean_name in self._object_to_part_token:
            return self._object_to_part_token[clean_name]
        for candidate_name in iter_name_variants(clean_name):
            part_token = self._object_to_part_token.get(candidate_name, "")
            if part_token:
                return part_token
        if len(self.part_layouts) == 1:
            return next(iter(self.part_layouts.keys()))
        return ""

    def _iter_draw_call_ranges(self, part_layout):
        vertex_cursor = 0
        for draw_call in part_layout.draw_calls:
            if draw_call.vertex_count:
                start_v = vertex_cursor
                end_v = vertex_cursor + int(draw_call.vertex_count) - 1
                vertex_cursor = end_v + 1
            else:
                start_v, end_v = self.config_node._calculate_vertex_range(draw_call.ib_path, draw_call.draw_params)
                if start_v is None or end_v is None:
                    continue
                vertex_cursor = max(vertex_cursor, int(end_v) + 1)
            yield draw_call, int(start_v), int(end_v)

    def _resolve_part_vertex_count_and_stride(self, part_layout, base_bytes: bytes) -> tuple[int, int]:
        total_draw_vertices = sum(
            int(draw_call.vertex_count)
            for draw_call in part_layout.draw_calls
            if draw_call.vertex_count
        )
        if total_draw_vertices > 0 and len(base_bytes) % total_draw_vertices == 0:
            return int(total_draw_vertices), int(len(base_bytes) / total_draw_vertices)

        first_draw_vertex_count = next(
            (int(draw_call.vertex_count) for draw_call in part_layout.draw_calls if draw_call.vertex_count),
            0,
        )
        if first_draw_vertex_count > 0 and len(base_bytes) % first_draw_vertex_count == 0:
            return int(first_draw_vertex_count), int(len(base_bytes) / first_draw_vertex_count)

        position_stride = 12 if len(base_bytes) % 12 == 0 else 16 if len(base_bytes) % 16 == 0 else 8
        vertex_count = int(len(base_bytes) / position_stride) if position_stride > 0 else 0
        return vertex_count, position_stride

    def _build_part_object_export_context_lookup(self, part_layout, d3d11_game_type):
        lookup = {}
        loop_index_cache = {}
        for draw_call, start_v, end_v in self._iter_draw_call_ranges(part_layout):
            export_indices = np.arange(start_v, end_v + 1, dtype=np.int32)
            mesh_obj = None
            for candidate_name in iter_name_variants(draw_call.mesh_name):
                mesh_obj = bpy.data.objects.get(candidate_name)
                if mesh_obj is not None:
                    break
            local_loop_indices = np.asarray([], dtype=np.int32)
            if mesh_obj is not None:
                cache_key = mesh_obj.name_full
                exported_loop_indices = loop_index_cache.get(cache_key)
                if exported_loop_indices is None:
                    exported_loop_indices = _build_exported_loop_indices(
                        mesh_obj,
                        exporter_module=self._modimp_exporter_module,
                        flip_uv_v=bool(getattr(self.exporter, "flip_uv_v", False)),
                        default_mirror_flip=bool(getattr(self.exporter, "default_mirror_flip", False)),
                    )
                    loop_index_cache[cache_key] = exported_loop_indices
                if exported_loop_indices.size > 0 and int(export_indices.max(initial=-1)) < exported_loop_indices.size:
                    local_loop_indices = exported_loop_indices[export_indices]
            context = {
                "export_indices": export_indices,
                "local_loop_indices": local_loop_indices,
                "d3d11_game_type": d3d11_game_type,
                "preferred_source_name": draw_call.mesh_name,
            }
            for candidate_name in iter_name_variants(draw_call.mesh_name):
                lookup.setdefault(candidate_name, context)
        return lookup

    def _collect_target_part_tokens(self) -> list[str]:
        part_tokens = []
        for multi_file_node in self.multi_file_nodes:
            for item in getattr(multi_file_node, "object_list", []) or []:
                part_token = self._resolve_part_token(getattr(item, "object_name", ""))
                if part_token and part_token not in part_tokens:
                    part_tokens.append(part_token)
        if part_tokens:
            return part_tokens
        return list(self.part_layouts.keys())

    def _build_runtime_infos(self, _hash_filters):
        runtime_infos = OrderedDict()
        for part_token in self._collect_target_part_tokens():
            part_layout = self.part_layouts.get(part_token)
            if part_layout is None:
                continue
            if not os.path.exists(part_layout.position_path):
                continue

            with open(part_layout.position_path, "rb") as file_obj:
                base_bytes = file_obj.read()

            vertex_count, position_stride = self._resolve_part_vertex_count_and_stride(part_layout, base_bytes)
            if vertex_count <= 0:
                raise MultiFileDirectExportError(f"NTMI MultiFile: invalid vertex count for {part_token}")

            d3d11_game_type = _build_minimal_position_game_type(position_stride)

            runtime_infos[part_token] = {
                "logical_hash": part_token,
                "actual_hash": f"{part_layout.file_stem}-multifile",
                "file_stem": part_layout.file_stem,
                "base_path": part_layout.position_path,
                "base_bytes": base_bytes,
                "position_stride": position_stride,
                "vertex_count": vertex_count,
                "drawib_model": None,
                "base_resource_name": part_layout.position_resource,
                "part_layout": part_layout,
                "object_export_context_lookup": self._build_part_object_export_context_lookup(
                    part_layout,
                    d3d11_game_type,
                ),
            }
        return runtime_infos

    def _build_object_entry_map(self, runtime_infos):
        entry_map = defaultdict(list)
        for part_token, runtime_info in runtime_infos.items():
            context_lookup = runtime_info.get("object_export_context_lookup", {}) or {}
            for candidate_name, context in context_lookup.items():
                export_indices = np.asarray(context.get("export_indices", []), dtype=np.int32)
                if export_indices.size == 0:
                    continue
                entry = {
                    "actual_hash": runtime_info["actual_hash"],
                    "export_indices": export_indices,
                    "local_loop_indices": np.asarray(context.get("local_loop_indices", []), dtype=np.int32),
                    "expected_bytes": int(export_indices.size) * runtime_info["position_stride"],
                    "d3d11_game_type": context.get("d3d11_game_type"),
                    "preferred_source_name": context.get("preferred_source_name", ""),
                    "part_token": part_token,
                }
                self._append_object_entry(entry_map, candidate_name, entry)
        return entry_map

    def _create_packed_position_state_buffers(self, base_bytes: bytes, target_bytes: bytes):
        base_meshes = np.frombuffer(base_bytes, dtype=np.float32)
        target_meshes = np.frombuffer(target_bytes, dtype=np.float32)
        min_len = min(len(base_meshes), len(target_meshes))
        min_len -= min_len % 3
        if min_len <= 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        base_positions = base_meshes[:min_len].reshape(-1, 3)
        target_positions = target_meshes[:min_len].reshape(-1, 3)
        delta_positions = target_positions - base_positions

        changed_mask = ~np.all(np.isclose(delta_positions, 0.0, atol=1e-6), axis=1)
        changed_indices = np.flatnonzero(changed_mask).astype(np.int32, copy=False)

        map_array = np.full(base_positions.shape[0], -1, dtype=np.int32)
        if changed_indices.size > 0:
            map_array[changed_indices] = np.arange(changed_indices.size, dtype=np.int32)

        packed_deltas = delta_positions[changed_mask].astype(np.float32, copy=False).reshape(-1)
        return map_array, packed_deltas

    def _format_position_bytes_from_coords(self, sampled_coords, position_stride=12):
        coords = np.asarray(sampled_coords, dtype=np.float32)
        if coords.size > 0:
            converted = np.empty_like(coords, dtype=np.float32)
            for index, (x_value, y_value, z_value) in enumerate(coords.tolist()):
                converted[index] = self._position_converter.from_blender_position(
                    (float(x_value), float(y_value), float(z_value))
                )
            coords = converted

        if int(position_stride) == 16:
            formatted = np.zeros((coords.shape[0], 4), dtype=np.float32)
            formatted[:, :3] = coords
            formatted[:, 3] = 1.0
            return formatted.tobytes()
        if int(position_stride) == 8:
            formatted = np.zeros((coords.shape[0], 4), dtype=np.float16)
            formatted[:, :3] = coords.astype(np.float16)
            formatted[:, 3] = 1.0
            return formatted.tobytes()
        return coords.astype(np.float32, copy=False).tobytes()

    def _build_object_position_bytes(self, obj, object_entry, position_stride: int, buffer_result_cache: dict | None = None):
        del buffer_result_cache

        local_loop_indices = np.asarray(object_entry.get("local_loop_indices", []), dtype=np.int32)
        if local_loop_indices.size == 0:
            return b""

        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_copy = self._modimp_exporter_module._evaluated_triangulated_mesh_copy(obj, depsgraph=depsgraph)
        try:
            all_loop_vertex_indices = np.empty(len(mesh_copy.loops), dtype=np.int32)
            mesh_copy.loops.foreach_get("vertex_index", all_loop_vertex_indices)
            max_loop_index = int(local_loop_indices.max()) if local_loop_indices.size > 0 else -1
            if max_loop_index >= len(all_loop_vertex_indices):
                raise MultiFileDirectExportError(
                    f"物体 '{obj.name}' 的导出 loop 映射越界: max_loop_index={max_loop_index}, loop_count={len(all_loop_vertex_indices)}"
                )

            sampled_vertex_indices = all_loop_vertex_indices[local_loop_indices]
            coords = np.empty((len(mesh_copy.vertices), 3), dtype=np.float32)
            mesh_copy.vertices.foreach_get("co", coords.ravel())
            sampled_coords = coords[sampled_vertex_indices]
            return self._format_position_bytes_from_coords(
                sampled_coords,
                position_stride=position_stride,
            )
        finally:
            bpy.data.meshes.remove(mesh_copy)

    def _build_state_outputs(self, runtime_info, object_entry_map, max_export_count):
        generated_states = OrderedDict()
        base_bytes = runtime_info["base_bytes"]
        object_buffer_cache = {}

        for export_index in range(2, max_export_count + 1):
            state_bytes = bytearray(base_bytes)
            state_changed = False

            for multi_file_node in self.multi_file_nodes:
                object_list = getattr(multi_file_node, "object_list", [])
                if not object_list:
                    continue

                base_item = object_list[0]
                target_item = object_list[min(export_index - 1, len(object_list) - 1)]
                if not getattr(base_item, "object_name", "") or not getattr(target_item, "object_name", ""):
                    continue
                if base_item.object_name == target_item.object_name:
                    continue

                any_base_entry = self._find_object_entry(object_entry_map, base_item)
                if any_base_entry is None:
                    raise MultiFileDirectExportError(f"无法定位多文件基础物体范围: {base_item.object_name}")

                base_entry = self._find_object_entry(
                    object_entry_map,
                    base_item,
                    actual_hash=runtime_info["actual_hash"],
                )
                if base_entry is None:
                    continue

                target_obj = bpy.data.objects.get(target_item.object_name)
                if target_obj is None:
                    raise MultiFileDirectExportError(f"找不到多文件目标物体: {target_item.object_name}")

                target_pos_bytes = self._build_object_position_bytes(
                    target_obj,
                    object_entry=base_entry,
                    position_stride=runtime_info["position_stride"],
                    buffer_result_cache=object_buffer_cache,
                )
                if len(target_pos_bytes) != base_entry["expected_bytes"]:
                    raise MultiFileDirectExportError(
                        f"多文件状态 {export_index}: 物体 '{target_item.object_name}' 顶点数/步长不一致，"
                        f"基础字节数={base_entry['expected_bytes']}，当前字节数={len(target_pos_bytes)}"
                    )

                self._apply_position_override(
                    state_bytes=state_bytes,
                    position_bytes=target_pos_bytes,
                    export_indices=np.asarray(base_entry.get("export_indices", []), dtype=np.int32),
                    position_stride=runtime_info["position_stride"],
                )
                state_changed = True

            if not state_changed or bytes(state_bytes) == base_bytes:
                continue

            map_array, pos_deltas_array = self._create_packed_position_state_buffers(
                base_bytes,
                bytes(state_bytes),
            )

            data_filename = f"{runtime_info['actual_hash']}-Position{export_index:02d}_packed_pos_delta.buf"
            map_filename = f"{runtime_info['actual_hash']}-Position{export_index:02d}_map.buf"
            data_path = os.path.join(self.meshes_dir, data_filename)
            map_path = os.path.join(self.meshes_dir, map_filename)
            self.config_node._write_Meshes_file(pos_deltas_array, data_path)
            self.config_node._write_Meshes_file(map_array, map_path)

            generated_states[export_index] = {
                "data_filename": data_filename,
                "map_filename": map_filename,
            }

        return generated_states

    def _ensure_present_run_lines(self, present_lines, run_lines):
        if not run_lines:
            return

        guard_line = f"if {self.config_node.active_swapkey} == {self.config_node.active_value}"
        active_block_start = -1
        active_block_end = -1
        nested_if_depth = 0

        for index, line in enumerate(present_lines):
            stripped_line = str(line or "").strip()
            if active_block_start < 0:
                if stripped_line == guard_line:
                    active_block_start = index
                    nested_if_depth = 1
                continue

            if stripped_line.startswith("if "):
                nested_if_depth += 1
            elif stripped_line == "endif":
                nested_if_depth -= 1
                if nested_if_depth == 0:
                    active_block_end = index
                    break

        if active_block_start >= 0 and active_block_end >= 0:
            existing_run_lines = {str(line or "").strip() for line in present_lines[active_block_start + 1:active_block_end]}
            insert_index = active_block_end
            for run_line in run_lines:
                if run_line.strip() in existing_run_lines:
                    continue
                present_lines.insert(insert_index, run_line)
                insert_index += 1
            return

        if present_lines and str(present_lines[-1] or "").strip():
            present_lines.append("")
        present_lines.append(guard_line)
        present_lines.extend(run_lines)
        present_lines.append("endif")

    def _multifile_shader_section_name(self, part_token: str) -> str:
        safe_token = self.config_node._hash_to_resource_prefix(part_token)
        return f"CustomShader_NTMI_MultiFile_{safe_token}"

    def _multifile_runtime_position_name(self, part_token: str) -> str:
        return f"ResourcePart_{part_token}_MultiFile_Position"

    def _multifile_runtime_position_uav_name(self, part_token: str) -> str:
        return f"ResourcePart_{part_token}_MultiFile_Position_UAV"

    def _multifile_resource_name(self, part_token: str, suffix: str) -> str:
        return f"ResourcePart_{part_token}_MultiFile_{suffix}"

    def _multifile_shader_filename(self, part_token: str) -> str:
        return f"multifile_anim_{part_token}.hlsl"

    def _write_multifile_shader_file(self, shader_path: str):
        shader_source = "\n".join(
            [
                "// NTMI pre-skin multifile position delta applier.",
                "// t51 = packed position delta buffer (float3)",
                "// t75 = vertex->packed index map",
                "// t54 = source position buffer (float triplets)",
                "// u5  = dynamic position output buffer (float triplets)",
                "",
                "StructuredBuffer<float3> delta_positions : register(t51);",
                "StructuredBuffer<int> delta_map : register(t75);",
                "Buffer<float> BasePosition : register(t54);",
                "RWBuffer<float> OutPosition : register(u5);",
                "",
                "[numthreads(64, 1, 1)]",
                "void main(uint3 threadID : SV_DispatchThreadID)",
                "{",
                "    uint vertex_id = threadID.x;",
                "    uint position_float_count = 0u;",
                "    BasePosition.GetDimensions(position_float_count);",
                "    uint vertex_count = position_float_count / 3u;",
                "    if (vertex_id >= vertex_count)",
                "    {",
                "        return;",
                "    }",
                "",
                "    uint position_base = vertex_id * 3u;",
                "    float3 position_value = float3(",
                "        BasePosition[position_base + 0u],",
                "        BasePosition[position_base + 1u],",
                "        BasePosition[position_base + 2u]",
                "    );",
                "",
                "    int packed_index = delta_map[vertex_id];",
                "    if (packed_index >= 0)",
                "    {",
                "        position_value += delta_positions[packed_index];",
                "    }",
                "",
                "    OutPosition[position_base + 0u] = position_value.x;",
                "    OutPosition[position_base + 1u] = position_value.y;",
                "    OutPosition[position_base + 2u] = position_value.z;",
                "}",
            ]
        )
        with open(shader_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(shader_source)

    def _patch_skin_commandlists(self, sections, runtime_infos):
        runtime_info_by_part = {
            str(part_token): runtime_info
            for part_token, runtime_info in runtime_infos.items()
        }

        active_variable = _normalized_active_variable(getattr(self.config_node, "active_swapkey", ""))
        animation_variable = str(getattr(self.config_node, "animation_swapkey", "") or "").strip() or "$swapkey100"
        try:
            active_value = int(getattr(self.config_node, "active_value", 1))
        except (TypeError, ValueError):
            active_value = 1

        for section_name, lines in list(sections.items()):
            if not str(section_name or "").startswith("[CommandList_SkinParts_"):
                continue

            current_part_token = ""
            patched_lines = []
            for line in lines:
                stripped = str(line or "").strip()

                if stripped.startswith("cs-t65 = ResourcePalette_"):
                    current_part_token = str(stripped.split("=", 1)[1] or "").strip().replace("ResourcePalette_", "")

                if stripped == "run = CommandList\\NTMIv1\\SkinFromBoundSlots" and current_part_token in runtime_info_by_part:
                    runtime_info = runtime_info_by_part[current_part_token]
                    runtime_position_resource = self._multifile_runtime_position_name(current_part_token)
                    runtime_position_uav = self._multifile_runtime_position_uav_name(current_part_token)
                    source_position_resource = ""
                    for previous_line in reversed(patched_lines):
                        previous_stripped = str(previous_line or "").strip()
                        if previous_stripped.startswith("cs-t68 = "):
                            source_position_resource = str(previous_stripped.split("=", 1)[1] or "").strip()
                            break
                    shader_section_name = self._multifile_shader_section_name(current_part_token)
                    patched_lines.extend(
                        [
                            f"if {active_variable} == {active_value} && {animation_variable} >= 2",
                            f"    cs-t54 = {source_position_resource or runtime_info['part_layout'].position_resource}",
                            f"    cs-u5 = {runtime_position_uav}",
                            f"    run = {shader_section_name}",
                            f"    {runtime_position_resource} = copy {runtime_position_uav}",
                            "    cs-t54 = null",
                            "    cs-u5 = null",
                            f"    cs-t68 = {runtime_position_resource}",
                            "endif",
                        ]
                    )
                    patched_lines.append(line)
                    continue

                patched_lines.append(line)

            sections[section_name] = patched_lines

    def _update_ini_sections(self, sections, preserved_tail_content, target_ini_file, runtime_infos, generated_states):
        dest_res_dir = os.path.join(self.mod_export_path, "res")
        os.makedirs(dest_res_dir, exist_ok=True)

        constants_section = "[Constants]"
        constants_lines = sections.get(constants_section, [])
        constants_content = "".join(constants_lines)
        if self.config_node.animation_swapkey not in constants_content:
            constants_lines.append(f"global persist {self.config_node.animation_swapkey} = 0")
        active_variable = _normalized_active_variable(getattr(self.config_node, "active_swapkey", ""))
        if active_variable not in constants_content:
            constants_lines.append(f"global persist {active_variable} = 0")
        generated_sections = OrderedDict()

        for part_token, runtime_info in runtime_infos.items():
            state_outputs = generated_states.get(part_token, {})
            if not state_outputs:
                continue

            part_layout = runtime_info["part_layout"]
            base_resource_name = part_layout.position_resource
            shader_section_name = self._multifile_shader_section_name(part_token)
            runtime_position_resource = self._multifile_runtime_position_name(part_token)
            runtime_position_uav = self._multifile_runtime_position_uav_name(part_token)

            shader_filename = self._multifile_shader_filename(part_token)
            shader_dest_path = os.path.join(dest_res_dir, shader_filename)
            self._write_multifile_shader_file(shader_dest_path)

            for export_index, state_info in state_outputs.items():
                data_section = f"[{self._multifile_resource_name(part_token, f'State{export_index:02d}_PackedPosDelta')}]"
                generated_sections[data_section] = [
                    "type = StructuredBuffer",
                    "stride = 12",
                    f"filename = Buffer/{state_info['data_filename']}",
                ]
                map_section = f"[{self._multifile_resource_name(part_token, f'State{export_index:02d}_Map')}]"
                generated_sections[map_section] = [
                    "type = StructuredBuffer",
                    "stride = 4",
                    f"filename = Buffer/{state_info['map_filename']}",
                ]

            position_float_count = int(runtime_info["vertex_count"]) * 3
            generated_sections[f"[{runtime_position_uav}]"] = [
                "dynamic_slots = 16",
                "type = RWBuffer",
                "format = R32_FLOAT",
                f"array = {position_float_count}",
            ]
            generated_sections[f"[{runtime_position_resource}]"] = [
                "dynamic_slots = 16",
                "type = Buffer",
                "format = R32_FLOAT",
                f"array = {position_float_count}",
            ]

            shader_lines = []
            if self.config_node.comment:
                shader_lines.append("; " + self.config_node.comment)
                shader_lines.append("")

            ordered_states = list(state_outputs.items())
            for export_index, _state_info in ordered_states:
                shader_lines.append(f"if {self.config_node.animation_swapkey} == {export_index}")
                shader_lines.append(
                    f"      cs-t51 = copy {self._multifile_resource_name(part_token, f'State{export_index:02d}_PackedPosDelta')}"
                )
                shader_lines.append("endif")
            shader_lines.append("")
            for export_index, _state_info in ordered_states:
                shader_lines.append(f"if {self.config_node.animation_swapkey} == {export_index}")
                shader_lines.append(
                    f"      cs-t75 = copy {self._multifile_resource_name(part_token, f'State{export_index:02d}_Map')}"
                )
                shader_lines.append("endif")

            shader_lines.append("")
            shader_lines.append(f"    cs = ./res/{shader_filename}")
            shader_lines.append(f"    dispatch = {runtime_info['vertex_count']}/64+1, 1, 1")
            shader_lines.append("    cs-t51 = null")
            shader_lines.append("    cs-t75 = null")
            generated_sections[f"[{shader_section_name}]"] = shader_lines

        self._patch_skin_commandlists(sections, runtime_infos)

        sections[constants_section] = constants_lines
        for section_name, lines in generated_sections.items():
            sections[section_name] = lines
        self.config_node._write_ordered_dict_to_ini(sections, target_ini_file, preserved_tail_content)

    def generate(self):
        self.config_node._create_cumulative_backup(self.target_ini_file, self.mod_export_path)
        runtime_infos = self._build_runtime_infos(None)
        if not runtime_infos:
            raise MultiFileDirectExportError("NTMI MultiFile: no valid NTMI part layout could be resolved.")

        object_entry_map = self._build_object_entry_map(runtime_infos)
        max_export_count = 1
        for multi_file_node in self.multi_file_nodes:
            max_export_count = max(max_export_count, len(getattr(multi_file_node, "object_list", []) or []))

        generated_states = {}
        for part_token, runtime_info in runtime_infos.items():
            generated_states[part_token] = self._build_state_outputs(runtime_info, object_entry_map, max_export_count)

        self._update_ini_sections(
            sections=self.sections,
            preserved_tail_content=self.preserved_tail_content,
            target_ini_file=self.target_ini_file,
            runtime_infos=runtime_infos,
            generated_states=generated_states,
        )


def execute_ntmi_multifile_postprocess(config_node, multi_file_nodes, output_dir: str, exporter):
    generator = NTMIDirectMultiFileGenerator(
        config_node=config_node,
        multi_file_nodes=multi_file_nodes,
        mod_export_path=output_dir,
        exporter=exporter,
    )
    generator.generate()
