"""Optional EFMI physics authoring with event-driven managed-bone cleanup."""

from . import operators, panel, properties, rig


def register():
    properties.register()
    operators.register()
    panel.register()
    rig.register_handlers()


def unregister():
    rig.unregister_handlers()
    panel.unregister()
    operators.unregister()
    properties.unregister()
