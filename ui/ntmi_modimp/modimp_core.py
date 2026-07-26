from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from pathlib import Path
from types import ModuleType


DEFAULT_MOD_IMPORTER_ROOT = Path(__file__).resolve().parents[3] / "mod_importer-main"
FALLBACK_MOD_IMPORTER_ROOT = Path(r"E:\代码\mod_importer-main")


@dataclass(frozen=True)
class ModImporterDependencyStatus:
    available: bool
    installed: bool
    root: str = ""
    source: str = ""
    message: str = ""
    checked_paths: tuple[str, ...] = ()


def _clean_configured_root(configured_root: str = "") -> str:
    return str(configured_root or "").strip().strip('"')


def _is_mod_importer_root(path: Path) -> bool:
    return (path / "core" / "exporter.py").is_file()


def _is_mod_importer_addon_module(module) -> bool:
    bl_info = getattr(module, "bl_info", {}) or {}
    addon_name = str(bl_info.get("name", "") or "").strip().lower()
    if addon_name == "mod importer":
        return True
    module_name = str(getattr(module, "__name__", "") or "").strip().lower()
    return "mod_importer" in module_name or "mod-importer" in module_name


def _module_root(module) -> Path | None:
    module_file = str(getattr(module, "__file__", "") or "").strip()
    if module_file:
        return Path(module_file).resolve().parent

    module_path = getattr(module, "__path__", None)
    if module_path:
        for item in module_path:
            if item:
                return Path(str(item)).resolve()
    return None


def _enabled_addon_module_names() -> set[str]:
    try:
        import bpy

        return set(getattr(bpy.context.preferences, "addons", {}).keys())
    except Exception:
        return set()


def _iter_installed_mod_importer_roots():
    try:
        import addon_utils
    except Exception:
        return

    try:
        modules = addon_utils.modules(refresh=False)
    except TypeError:
        modules = addon_utils.modules()
    except Exception:
        return

    enabled_names = _enabled_addon_module_names()
    for module in modules:
        if not _is_mod_importer_addon_module(module):
            continue
        root = _module_root(module)
        if root is None:
            continue
        module_name = str(getattr(module, "__name__", "") or "")
        yield root, module_name in enabled_names


def _candidate_roots(configured_root: str = "") -> list[tuple[str, Path, bool]]:
    candidates: list[tuple[str, Path, bool]] = []
    clean_configured_root = _clean_configured_root(configured_root)
    if clean_configured_root:
        candidates.append(("手动路径", Path(clean_configured_root).expanduser(), False))

    for addon_root, enabled in _iter_installed_mod_importer_roots() or []:
        candidates.append(("已启用前置插件" if enabled else "已安装前置插件", addon_root, True))

    candidates.append(("相邻源码目录", DEFAULT_MOD_IMPORTER_ROOT, False))
    candidates.append(("默认本地目录", FALLBACK_MOD_IMPORTER_ROOT, False))
    return candidates


def detect_mod_importer_dependency(configured_root: str = "") -> ModImporterDependencyStatus:
    checked_paths = []
    seen = set()

    for source, candidate, installed in _candidate_roots(configured_root):
        root = candidate.resolve()
        root_key = str(root).lower()
        if root_key in seen:
            continue
        seen.add(root_key)
        checked_paths.append(f"{source}: {root}")
        if _is_mod_importer_root(root):
            message = f"已检测到前置插件（{source}）" if installed else f"已找到可用前置插件目录（{source}）"
            return ModImporterDependencyStatus(
                available=True,
                installed=installed,
                root=str(root),
                source=source,
                message=message,
                checked_paths=tuple(checked_paths),
            )

    return ModImporterDependencyStatus(
        available=False,
        installed=False,
        message="未检测到 mod_importer-main 前置插件或可用源码目录",
        checked_paths=tuple(checked_paths),
    )


def resolve_mod_importer_root(configured_root: str = "") -> Path:
    status = detect_mod_importer_dependency(configured_root)
    if status.available:
        return Path(status.root)

    checked = "; ".join(status.checked_paths)
    raise FileNotFoundError(
        "mod_importer-main core exporter was not found. "
        f"Checked: {checked}"
    )


def ensure_mod_importer_package(configured_root: str = "") -> ModuleType:
    root = resolve_mod_importer_root(configured_root)
    package_name = "theherta4_external_mod_importer"
    root_path = str(root)

    loaded = sys.modules.get(package_name)
    if loaded is not None:
        if list(getattr(loaded, "__path__", [])) == [root_path]:
            return loaded
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]

    module = ModuleType(package_name)
    module.__file__ = str(root / "__init__.py")
    module.__path__ = [root_path]
    module.__package__ = package_name
    sys.modules[package_name] = module
    return module


def get_export_collection_package(configured_root: str = ""):
    package = ensure_mod_importer_package(configured_root)
    exporter_module = importlib.import_module(f"{package.__name__}.core.exporter")
    return exporter_module.export_collection_package
