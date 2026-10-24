import os

import bpy

from .node_base import SSMTNodeBase


MASK_NONE = 'NONE'
MASK_DIFFUSE_A = 'DIFFUSE_A'
MASK_TEXTURE = 'TEXTURE'

# R/G/B 读遮罩贴图的对应通道，A 读本部件漫射贴图的 A 通道；同一时间只选一个。
_TEXTURE_CHANNEL_PROPS = ("use_mask_r", "use_mask_g", "use_mask_b")
_CHANNEL_PROPS = _TEXTURE_CHANNEL_PROPS + ("use_diffuse_alpha",)


def _keep_one_channel(prop_name):
    def update(self, context):
        if not getattr(self, prop_name):
            return
        for other in _CHANNEL_PROPS:
            if other != prop_name and getattr(self, other):
                setattr(self, other, False)
    return update


class SSMTNode_Translucency(SSMTNodeBase):
    '''用 EFMI TTL 半透明库绘制连到它的物体（仅 EFMI 骨骼合并导出）'''
    bl_idname = 'SSMTNode_Translucency'
    bl_label = '半透明'
    bl_icon = 'MOD_OPACITY'

    alpha: bpy.props.FloatProperty(
        name="不透明度",
        default=0.5,
        min=0.0,
        max=1.0,
        precision=2,
        subtype='FACTOR',
    )

    use_mask_r: bpy.props.BoolProperty(
        name="R",
        description="用遮罩贴图的 R 通道",
        default=False,
        update=_keep_one_channel("use_mask_r"),
    )

    use_mask_g: bpy.props.BoolProperty(
        name="G",
        description="用遮罩贴图的 G 通道",
        default=False,
        update=_keep_one_channel("use_mask_g"),
    )

    use_mask_b: bpy.props.BoolProperty(
        name="B",
        description="用遮罩贴图的 B 通道",
        default=False,
        update=_keep_one_channel("use_mask_b"),
    )

    use_diffuse_alpha: bpy.props.BoolProperty(
        name="A",
        description="用漫射贴图的 A 通道",
        default=False,
        update=_keep_one_channel("use_diffuse_alpha"),
    )

    mask_texture_path: bpy.props.StringProperty(
        name="遮罩贴图",
        subtype='FILE_PATH',
        default="",
    )

    def init(self, context):
        self.inputs.new('SSMTSocketObject', "Input 1")
        self.outputs.new('SSMTSocketObject', "Output")
        self.width = 200

    def draw_buttons(self, context, layout):
        # 选了通道时只按遮罩决定，不透明度不写出。
        row = layout.row()
        row.enabled = not any(getattr(self, prop) for prop in _CHANNEL_PROPS)
        row.prop(self, "alpha")
        row = layout.row(align=True)
        for prop in _CHANNEL_PROPS:
            row.prop(self, prop, toggle=True)
        if self.get_texture_channel() is not None:
            layout.prop(self, "mask_texture_path", text="")

    def update(self):
        if self.inputs and self.inputs[-1].is_linked:
            self.inputs.new('SSMTSocketObject', f"Input {len(self.inputs) + 1}")

        if len(self.inputs) > 1 and not self.inputs[-1].is_linked and not self.inputs[-2].is_linked:
            self.inputs.remove(self.inputs[-1])

    def get_texture_channel(self):
        for channel, prop in enumerate(_TEXTURE_CHANNEL_PROPS):
            if getattr(self, prop):
                return channel
        return None

    def get_translucency(self) -> dict:
        translucency = {"alpha": None, "channel": None, "texture": "", "node": self.name}
        texture_channel = self.get_texture_channel()
        if self.use_diffuse_alpha:
            translucency.update(mask=MASK_DIFFUSE_A, channel=3)
        elif texture_channel is not None:
            texture = self.mask_texture_path.strip()
            translucency.update(
                mask=MASK_TEXTURE,
                channel=texture_channel,
                texture=os.path.normpath(bpy.path.abspath(texture)) if texture else "",
            )
        else:
            translucency.update(mask=MASK_NONE, alpha=round(float(self.alpha), 3))
        return translucency

    @staticmethod
    def generate_signature(node) -> str:
        translucency = SSMTNode_Translucency.get_translucency(node)
        if translucency["mask"] == MASK_NONE:
            return f"Translucency[alpha={translucency['alpha']}]"
        if translucency["mask"] == MASK_DIFFUSE_A:
            return f"Translucency[mask={MASK_DIFFUSE_A}]"
        return (
            f"Translucency[mask={MASK_TEXTURE},channel={'RGB'[translucency['channel']]},"
            f"texture={translucency['texture']}]"
        )


classes = (
    SSMTNode_Translucency,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
