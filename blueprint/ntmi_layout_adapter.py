from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import os
import re
from typing import Iterable


_RESOURCE_PART_POSITION_RE = re.compile(r"^\[ResourcePart_(?P<token>.+)_Position\]$")
_RESOURCE_PART_IB_RE = re.compile(r"^\[ResourcePart_(?P<token>.+)_IB\]$")
_TEXTURE_OVERRIDE_RE = re.compile(r"^\[TextureOverride_IB_(?P<name>.+)\]$")
_PART_COMMENT_RE = re.compile(r"^\s*;\s*\[part:(?P<name>[^\]]+)\]")
_MESH_COMMENT_RE = re.compile(
    r"^\s*;\s*\[mesh:(?P<name>[^\]]+)\](?:\s+\[vertex_count:(?P<vertex_count>\d+)\])?"
)
_PART_TOKEN_FROM_RESOURCE_RE = re.compile(r"^ResourcePart_(?P<token>.+)_IB$")
_DRAW_IB_PREFIX_RE = re.compile(r"^(?P<draw_ib>[0-9a-fA-F]{8})")


@dataclass
class NTMIDrawCallLayout:
    mesh_name: str
    part_token: str
    part_name: str
    position_resource: str
    ib_resource: str
    position_path: str
    ib_path: str
    draw_params: tuple[int, int, int]
    vertex_count: int | None = None
    start_vertex: int | None = None
    end_vertex: int | None = None


@dataclass
class NTMIPartLayout:
    part_token: str
    part_name: str
    draw_ib: str
    position_resource: str
    ib_resource: str
    position_path: str
    ib_path: str
    file_stem: str
    source_ini_path: str
    draw_calls: list[NTMIDrawCallLayout] = field(default_factory=list)


def iter_name_variants(name: str) -> Iterable[str]:
    clean_name = str(name or "").strip()
    if not clean_name:
        return []

    variants = OrderedDict()
    variants[clean_name] = True

    if clean_name.endswith("_copy"):
        variants[clean_name[:-5]] = True

    stripped = re.sub(r"(_(?:chain|dup|copy|BPE)\d*)+$", "", clean_name, flags=re.IGNORECASE)
    if stripped:
        variants[stripped] = True

    stripped_x = re.sub(r"_x\d+$", "", stripped)
    if stripped_x:
        variants[stripped_x] = True

    return variants.keys()


def normalize_resource_path(base_dir: str, resource_filename: str) -> str:
    filename = str(resource_filename or "").strip().replace("/", os.sep)
    if not filename:
        return ""
    return os.path.normpath(os.path.join(base_dir, filename))


def parse_draw_command(stripped_line: str) -> tuple[int, int, int] | None:
    lowered = stripped_line.lower()
    if lowered.startswith("drawindexed "):
        try:
            parts = [part.strip() for part in stripped_line.split("=", 1)[1].split(",")]
        except IndexError:
            return None
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None

    if lowered.startswith("drawindexedinstanced "):
        try:
            parts = [part.strip() for part in stripped_line.split("=", 1)[1].split(",")]
        except IndexError:
            return None
        if len(parts) < 5:
            return None
        try:
            return int(parts[0]), int(parts[2]), int(parts[3])
        except ValueError:
            return None

    return None


def draw_ib_from_part_name(part_name: str) -> str:
    match = _DRAW_IB_PREFIX_RE.match(str(part_name or "").strip())
    if not match:
        return ""
    return match.group("draw_ib").lower()


def _part_token_from_ib_resource(resource_name: str) -> str:
    match = _PART_TOKEN_FROM_RESOURCE_RE.match(str(resource_name or "").strip())
    if not match:
        return ""
    return str(match.group("token") or "").strip()


def _resource_filename(lines: list[str]) -> str:
    for line in lines:
        stripped = str(line or "").strip()
        if stripped.lower().startswith("filename ="):
            return str(stripped.split("=", 1)[1] or "").strip()
    return ""


def parse_ntmi_part_layouts(
    sections: "OrderedDict[str, list[str]]",
    *,
    output_dir: str,
    source_ini_path: str,
) -> "OrderedDict[str, NTMIPartLayout]":
    resource_to_filename: dict[str, str] = {}
    parts: "OrderedDict[str, NTMIPartLayout]" = OrderedDict()

    for section_name, lines in sections.items():
        resource_name = str(section_name or "").strip().strip("[]")
        filename = _resource_filename(lines)
        if filename:
            resource_to_filename[resource_name] = filename

    for section_name, lines in sections.items():
        if not _TEXTURE_OVERRIDE_RE.match(section_name):
            continue

        current_part_name = ""
        current_mesh_name = ""
        current_mesh_vertex_count: int | None = None
        current_ib_resource = ""

        for line in lines:
            stripped = str(line or "").strip()
            if not stripped:
                continue

            part_match = _PART_COMMENT_RE.match(stripped)
            if part_match:
                current_part_name = str(part_match.group("name") or "").strip()
                continue

            if stripped.lower().startswith("ib ="):
                current_ib_resource = str(stripped.split("=", 1)[1] or "").strip()
                continue

            mesh_match = _MESH_COMMENT_RE.match(stripped)
            if mesh_match:
                current_mesh_name = str(mesh_match.group("name") or "").strip()
                raw_vertex_count = str(mesh_match.group("vertex_count") or "").strip()
                current_mesh_vertex_count = int(raw_vertex_count) if raw_vertex_count else None
                continue

            draw_params = parse_draw_command(stripped)
            if draw_params is None or not current_mesh_name or not current_ib_resource:
                continue

            part_token = _part_token_from_ib_resource(current_ib_resource)
            if not part_token:
                current_mesh_name = ""
                current_mesh_vertex_count = None
                continue

            position_resource = f"ResourcePart_{part_token}_Position"
            position_filename = resource_to_filename.get(position_resource, "")
            ib_filename = resource_to_filename.get(current_ib_resource, "")
            if not position_filename or not ib_filename:
                current_mesh_name = ""
                current_mesh_vertex_count = None
                continue

            part_name = current_part_name or Path(ib_filename).name.replace("-ib.buf", "")
            draw_ib = draw_ib_from_part_name(part_name)
            part_layout = parts.get(part_token)
            if part_layout is None:
                part_layout = NTMIPartLayout(
                    part_token=part_token,
                    part_name=part_name,
                    draw_ib=draw_ib,
                    position_resource=position_resource,
                    ib_resource=current_ib_resource,
                    position_path=normalize_resource_path(output_dir, position_filename),
                    ib_path=normalize_resource_path(output_dir, ib_filename),
                    file_stem=Path(position_filename).name.replace("-position.buf", ""),
                    source_ini_path=source_ini_path,
                )
                parts[part_token] = part_layout

            part_layout.draw_calls.append(
                NTMIDrawCallLayout(
                    mesh_name=current_mesh_name,
                    part_token=part_token,
                    part_name=part_layout.part_name,
                    position_resource=position_resource,
                    ib_resource=current_ib_resource,
                    position_path=part_layout.position_path,
                    ib_path=part_layout.ib_path,
                    draw_params=draw_params,
                    vertex_count=current_mesh_vertex_count,
                )
            )
            current_mesh_name = ""
            current_mesh_vertex_count = None

    return parts
