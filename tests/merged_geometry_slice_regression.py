# -*- coding: utf-8 -*-
'''Merged Skeleton 共享 IB/VB 切片回归测试。'''

import importlib
import struct
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy


def _load_extractor_module():
    root = Path(__file__).resolve().parent.parent
    package_name = "_loyaltools_geometry_regression"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".extract.dump_workspace_extractor")


def main():
    module = _load_extractor_module()

    with tempfile.TemporaryDirectory(prefix="loyal-merged-geometry-") as folder:
        folder = Path(folder)
        ib_path = folder / "shared.buf"
        txt_path = folder / "draw-ib.txt"

        # 模拟共享 IB: 当前组件的原始索引是 454..456，
        # 但 EFMI ObjectExtractor 已把资源对象上的内存 buffer 改成 0..2。
        first_index = 6
        prefix = b"\0" * (32 + first_index * 2)
        ib_path.write_bytes(prefix + struct.pack("<3H", 454, 455, 456))
        txt_path.write_text(
            "\n".join(
                (
                    "byte offset: 32",
                    "first index: 6",
                    "index count: 3",
                    "topology: trianglelist",
                    "format: DXGI_FORMAT_R16_UINT",
                    "",
                    "454 455 456",
                )
            ),
            encoding="utf-8",
        )

        class MutatedBuffer:
            @staticmethod
            def get_field(_semantic):
                return numpy.array([0, 1, 2], dtype=numpy.uint16)

        ib = SimpleNamespace(
            hash="7ea509b0",
            format=module.DXGI_FORMAT.DXGI_FORMAT_R16_UINT,
            parent=None,
            byte_offset=0,
            txt_path=txt_path,
            txt_path_deduped=None,
            bin_path=ib_path,
            bin_path_deduped=None,
            topology=module.Topology.TriangleList,
            data_descriptor=None,
            migoto_format=None,
            buffer=MutatedBuffer(),
        )
        draw = SimpleNamespace(first_index=first_index, index_count=3, first_vertex=0)
        record = module._DrawRecord(shader_call=SimpleNamespace(), draw_call=draw, ib=ib)

        extractor = object.__new__(module.DumpWorkspaceExtractor)
        extractor.verbose = False
        extractor._txt_format_cache = {}
        indices, vertex_offset, vertex_count = extractor._build_index_data(record)

        assert indices.tolist() == [0, 1, 2], indices.tolist()
        assert vertex_offset == 454, vertex_offset
        assert vertex_count == 3, vertex_count
        print("MERGED_GEOMETRY_SLICE_REGRESSION=PASS")


if __name__ == "__main__":
    main()
