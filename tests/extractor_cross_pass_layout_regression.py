"""Regression for selecting the most complete VB layout across render passes."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path


def _load_modules():
    addon_root = Path(__file__).resolve().parents[1]
    package_name = "_loyaltools_cross_pass_layout_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(addon_root)]
    sys.modules[package_name] = package
    extractor_module = importlib.import_module(
        package_name + ".extract.dump_workspace_extractor"
    )
    byte_buffer_module = importlib.import_module(
        package_name + ".efmi_extract.migoto_io.data_model.byte_buffer"
    )
    format_module = importlib.import_module(
        package_name + ".efmi_extract.migoto_io.data_model.dxgi_format"
    )
    migoto_format_module = importlib.import_module(
        package_name + ".efmi_extract.migoto_io.migoto_model.migoto_format"
    )
    resources_module = importlib.import_module(
        package_name + ".efmi_extract.migoto_io.migoto_model.frame_model.resources"
    )
    return (
        extractor_module,
        byte_buffer_module,
        format_module,
        migoto_format_module,
        resources_module,
    )


def _make_format(byte_buffer, formats, migoto_format, include_tangent: bool):
    semantics = [
        byte_buffer.BufferSemantic(
            byte_buffer.AbstractSemantic(byte_buffer.Semantic.Position, 0),
            formats.DXGIFormat.R32G32B32_FLOAT,
            offset=0,
            input_slot=0,
        ),
        byte_buffer.BufferSemantic(
            byte_buffer.AbstractSemantic(byte_buffer.Semantic.Normal, 0),
            formats.DXGIFormat.R32G32B32_FLOAT,
            offset=12,
            input_slot=0,
        ),
    ]
    if include_tangent:
        semantics.append(
            byte_buffer.BufferSemantic(
                byte_buffer.AbstractSemantic(byte_buffer.Semantic.Tangent, 0),
                formats.DXGIFormat.R32G32B32A32_FLOAT,
                offset=24,
                input_slot=0,
            )
        )
    layout = byte_buffer.BufferLayout(
        semantics=semantics,
        auto_offsets=False,
        auto_stride=False,
    )
    layout.stride = 40
    return migoto_format.MigotoFormat(
        byte_offset=0,
        stride=40,
        first_vertex=0,
        vertex_count=2,
        vb_layout=layout,
    )


def main() -> None:
    extractor_module, byte_buffer, formats, migoto_format, resources = _load_modules()

    incomplete = _make_format(
        byte_buffer, formats, migoto_format, include_tangent=False
    )
    complete = _make_format(
        byte_buffer, formats, migoto_format, include_tangent=True
    )

    primary_vb = resources.VertexBuffer(hash="9178754a", pointer="primary")
    material_vb = resources.VertexBuffer(hash="9178754a", pointer="material")
    format_by_pointer = {
        primary_vb.pointer: incomplete,
        material_vb.pointer: complete,
    }

    class _Resources:
        def __init__(self, vb):
            self.vb = vb

        def get_by_slot(self, slot):
            return self.vb if slot == "vb0" else None

    records = [
        types.SimpleNamespace(
            shader_call=types.SimpleNamespace(
                id=22, model_resources=_Resources(primary_vb)
            )
        ),
        types.SimpleNamespace(
            shader_call=types.SimpleNamespace(
                id=47, model_resources=_Resources(material_vb)
            )
        ),
    ]

    extractor = object.__new__(extractor_module.DumpWorkspaceExtractor)
    extractor.verbose = False
    extractor._semantic_remap = extractor_module._build_semantic_remap()
    extractor._load_txt_format = lambda vb: format_by_pointer[vb.pointer]

    selected = extractor._select_slot_layout_format(records, primary_vb, 0)
    assert selected is complete
    assert extractor._score_slot_layout(incomplete, 0)[1] == 24
    assert extractor._score_slot_layout(complete, 0)[1] == 40

    # Overlapping aliases do not cover additional bytes. They must not replace
    # the first equally complete pass merely because it declares more entries.
    alias_layout = byte_buffer.BufferLayout(
        semantics=list(complete.vb_layout.semantics)
        + [
            byte_buffer.BufferSemantic(
                byte_buffer.AbstractSemantic(byte_buffer.Semantic.TexCoord, 7),
                formats.DXGIFormat.R32G32B32A32_FLOAT,
                offset=24,
                input_slot=0,
            )
        ],
        auto_offsets=False,
        auto_stride=False,
    )
    alias_layout.stride = 40
    alias_format = migoto_format.MigotoFormat(
        stride=40, vertex_count=2, vb_layout=alias_layout
    )
    assert extractor._score_slot_layout(alias_format, 0) == extractor._score_slot_layout(
        complete, 0
    )

    raw = bytes(range(80))
    with tempfile.TemporaryDirectory(prefix="loyal_cross_pass_") as temp_folder:
        raw_path = Path(temp_folder) / "position.buf"
        raw_path.write_bytes(raw)
        primary_vb.bin_path = raw_path
        slot_data, unknown_offset = extractor._build_slot_data(
            vb=primary_vb,
            slot_id=0,
            vertex_offset=0,
            vertex_count=2,
            unknown_index_offset=0,
            warnings=[],
            layout_format=selected,
        )

    names = [semantic.abstract.enum for semantic in slot_data.layout.semantics]
    assert byte_buffer.Semantic.Tangent in names
    assert byte_buffer.Semantic.Unknown not in names
    assert slot_data.raw_bytes == raw
    assert unknown_offset == 0
    print("EXTRACTOR_CROSS_PASS_LAYOUT_REGRESSION=PASS")


if __name__ == "__main__":
    main()
