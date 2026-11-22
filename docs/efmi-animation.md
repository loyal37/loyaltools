# EFMI bone animation

## Authoring in Blender

1. Open an EFMI **Merged Skeleton** workspace and select a mesh that already has
   native merged-skeleton weights. In the 3D Viewport sidebar (`N`), open
   **LoyalTools > EFMI Bone Animation**, turn on **Enable EFMI Bone Animation**,
   and click **Add Selected Mesh**.
2. Click **Add Animation Root**. Use the **Native Bone** selector to choose the
   bone the branch will follow in the game, then click **Edit Bone Rest Pose**
   to adjust the animation bone. To branch the rig, select any bone in the list
   and click **Add Child Bone**.
3. Click **Select Vertex Group** to activate the bone's `EFMI_ANIM_` vertex group
   for weight painting. Click **Select Pose Bone** to enter Pose Mode, then press
   `K` in the 3D Viewport and choose **Location & Rotation** to insert keyframes.
   Keyframe each root and child independently. For a full turn, animate one axis
   from 0° to 360° in the default XYZ Euler mode. Set the F-curve interpolation to
   **Linear** for a constant rotation speed.
4. Set **Start Frame**, **End Frame**, and **Playback Speed** for each root. Its
   children share the root's playback timing and have their own keyframes and
   local transforms. Export through the existing EFMI Merged Skeleton blueprint
   workflow.

The exporter samples both endpoints. When the start and end poses match, the
clip loops directly to the start. Otherwise it interpolates back to the start
over one frame. Use the same start and end frame to export a static pose. Check
the in-game skinning wherever vertices blend weights from animation bones with
native or physics bones.

In an exported mod with no physics chains attached to animation bones, press
**Ctrl+Alt+F8** to toggle animation. When the export includes both animation and
physics, press **Ctrl+Alt+F7** to toggle physics simulation. Change the shortcuts
in the exported mod INI if needed.

## Attaching physics chains to animation bones

1. Enable both modules and register the same EFMI mesh in the bone animation and
   physics panels.
2. Select a physics chain in the physics panel. Open the attachment selector and
   select a root or child **Animation Bone** from that mesh. The selected bone
   becomes the chain's sole attachment.
3. In the physics armature's Edit Mode, adjust the chain root's rest position
   relative to the animation bone. As the animation bone moves or rotates, the
   chain root follows and the chain simulates with its existing settings.

To disable bone animation in Blender, first choose a native attachment for each
physics chain that follows an animation bone. Mod exports with these attachments
keep animation playback enabled. When you turn off physics simulation in the
game, the attached parts continue to follow their selected animation bones rigidly.

## Export and runtime scope

- Supports Blender 4.5 or later and EFMI Merged Skeleton's GPU rolling-buffer
  path. Resource binding and release were checked against EFMI 1.5.0 and XXMI
  1.3.3. When a draw path cannot bind the extended bone palette, animation
  bones and any attached physics chains follow the resolved native bone
  rigidly. Other meshes sharing the same index buffer are still drawn.
- Each export supports up to 64 animation bones and 131,072 bone samples. Physics
  bones have their own limit. The exporter samples eight times per Blender frame.
  If a bone rotates more than 180° between samples, spread the rotation over more
  Blender frames.
- Keyframe pose bones and keep the armature object's transform static. Export
  reports any local shear or zero scale. To export a mesh with nonzero animation
  weights, enable bone animation or adjust the weights first.
- After switching to the correct workspace, continue using the existing
  animation rig. Export resolves each root's saved native bone anchor against
  the active workspace. If an anchor is invalid, choose a valid native bone in
  the animation panel. This preserves the existing weights, keyframes, and
  physics chains attached to the animation bones.
- Export checks the root's native bone source at the levels where its weighted
  meshes or attached physics chains are drawn. It can select an equivalent
  slot from another component without changing the authored follow group.
