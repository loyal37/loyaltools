# EFMI physics bones

This optional module adds editable physics chains to the EFMI Merged Skeleton
workflow. It uses the existing full-detail and LoD capture profile. Standard
export, ordinary Cross IB, and ZZMI do not use this module.

## Authoring

1. Extract/import an EFMI **Merged Skeleton** workspace and add its LoD mappings
   using the existing LoyalTools workflow.
2. Finish the mesh's original merged-skeleton weights first. The physics panel
   shows this prerequisite as a warning. A chain follows an original numeric
   group, whose matrix must be supplied by the captured character.
3. Open **LoyalTools > EFMI Physics Bones** and enable **Enable EFMI Physics**. This is
   the master switch, disabled by default and saved in the blend file. Select a
   mesh in Object Mode and choose **Add Selected Object**. Repeat for additional meshes.
4. Add a physics chain. It starts with one real parent bone; **Append Child Bone** adds
   connected segments. A parent-only chain is supported. Select the original
   group with **Choose Original Vertex Group**. Multiple chains can use different original groups.
5. Use **Edit Physics Armature** to enter the real armature's Edit Mode. Edit heads, tails,
   lengths, and roll. The root head is the attachment point. Keep child segments
   connected. Use Edit Mode rather than Pose Mode animation or bone constraints.
6. Return to the mesh and paint the prefixed groups displayed beside each segment.
   Every segment owns a stable name such as `EFMI_PHYS_000001`, separate from
   original numeric groups. Changing workspaces does not rename these groups or
   move their weights; export maps them after the current workspace's original B.
   Merely creating bones does not transfer weights automatically.
7. Adjust damping, restoring stiffness, maximum swing angle, model-space gravity,
   and wind. **Model-space Gravity** is the post-skinning game model space, not Blender's
   editing axes or world gravity. Its default is `(0, -9.8, 0)` and can be changed.
8. With the master switch enabled, run the panel's validation and use the existing
   **Generate Mod** workflow. The exporter emits the physics shaders and profile
   alongside the usual mod files.

Removing a chain or a segment with the panel removes its owned groups and weights,
without changing original groups or renaming surviving groups. The segment
button explicitly removes the selected child and its descendants. Authoring IDs
are not recycled during the scene's authoring session.

Deleting managed bones with Blender's native Edit Mode delete command also
cleans their owned groups after leaving Edit Mode. Surviving bones retain their
IDs and weights. If deleting an interior segment leaves a gap, reconnect/reposition
the surviving bones before export. Undo restores the bones, records, and weights.
The dependency-graph callback only queues managed updates. While enabled, a
one-second service synchronizes registered objects, including newly extruded
bones and missing owned groups. Removing an object clears its ownership markers
and excludes it from subsequent synchronization. Unregistered groups are never
claimed just because their names are numeric or contain the prefix.

Disabling the master switch stops authoring operators, automatic synchronization,
and physics export collection, while retaining the authoring data. Its on/off
state persists through save/reopen. Current-workspace metadata refreshes on enable,
before collection, and during enabled synchronization; it is not permanently
locked by an earlier workspace selection. Verified legacy numeric physics groups
are renamed to the dedicated prefix on enable without changing their weights.

The armature is editing data and is not automatically added as a mesh modifier.
If a user adds a preview Armature modifier for this managed rig, enabled EFMI
preprocessing removes it only from the temporary export copy, preventing the
game from deforming already-deformed vertices. The original object and unrelated
armature modifiers are preserved.

## Export and runtime contract

- Original component counts, private pools, and LoD remaps remain unchanged.
  Custom bones are appended to a separate character-wide extended palette.
  All participating GPU component callbacks bind that same palette, so custom
  bones are available across IBs.
- Stable prefixed authoring group names are explicitly mapped to compact runtime IDs.
  The exporter does not assume Blender's mutable group index equals the numeric
  group name. Deleting groups therefore does not redirect surviving weights.
  `res/EFMIPhysics/authoring.json` records the mapping and edited rest transforms.
- New bones do not pass through the original component-private/LoD bone remaps.
  The existing exporter still generates the per-LoD vertex layouts, including
  16-bit custom indices.
- Each chain has independent original-anchor freshness and pending simulation.
  A frame publishes the preceding pending result once, then freezes current and
  previous-render matrices for every consuming pass. This adds one render frame
  of physics latency and prevents shadow/color passes from integrating twice.
- Root heads follow original skinning matrices. Child heads follow their parent's
  tail. The solver applies length and swing constraints, damping, gravity, wind,
  and restoring stiffness. Each edited segment contributes its own skin matrix
  using its exported inverse bind.
- Temporary native resource references are restored with view caching disabled,
  then explicitly replaced and cleared. Physics buffers do not depend on viewport
  size and do not borrow render targets or backbuffers.
- An unsupported runtime source uses a rigid-follow fallback Blend buffer rather
  than indexing custom bones in an unextended palette. Known unsupported source
  profiles and invalid authoring data fail preflight with an error.
- With the option disabled or no physical bones configured, no physics resources
  or runtime are emitted and the original export path is retained. If a managed
  mesh still has nonzero physics-group weights while disabled, export reports an
  error before changing existing output files. It does not silently reinterpret
  those weights as native bones; enable the module or explicitly restore native
  weights first. Already-exported game mods only change after a successful full
  export/reload; buffer-only export does not replace the old INI.

## Current scope

Blender 4.5+ and EFMI Merged Skeleton's rolling-buffer posing path are supported.
The current bounded GPU implementation supports up to 64 additional segments per
character and 32 instance slots, subject to the 16-bit total bone-index limit.
The solver works in model space. World inertia, collision capsules, self-collision,
Blender Pose Mode animation, branching child graphs, and CB-with-bones posing
adapters are not implemented by this feature.

Objects with parenting/constraints and animated or constrained authoring rigs
are rejected instead of silently baking an ambiguous transform. Apply/resolve
those authoring dependencies before configuring physics. Edited rest-frame roll
and non-identity object transforms are exported with the normal EFMI coordinate
and non-mirror workflow.

## Validation

Pure Python export/runtime checks:

```sh
python -m unittest discover -s tests
```

On Windows, the GPU tests compile the shaders with D3DCompiler 47 and dispatch
them on a headless D3D11 WARP device. Blender tests are skipped by ordinary Python;
run them in Blender:

```sh
blender --background --factory-startup --python-exit-code 1 --python tests/test_efmi_physics_rig.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_efmi_physics_preprocess.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_efmi_physics_integration.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_efmi_physics_workspace.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_efmi_physics_nonmirror.py
```

Tests cover multi-object/multi-chain authoring, parent-only and multi-segment
chains, numeric ID stability, Edit Mode changes, save/reopen, deletion and Undo,
coordinate transforms, LoD layout conversion, fallback buffers, default-path
no-ops, and atomic INI replacement. Real GPU tests cover two chains with two
segments on two instance slots, constraints, freshness, per-frame freezing,
history, resets, and config offsets. They do not replace game-specific acceptance
tests of drawing order, LoD transitions, resolution changes, or fullscreen changes.
The full Blender integration fixture creates two meshes and two two-segment chains
and runs the actual EFMI buffer/INI exporter with the feature disabled and enabled,
including LoD1 layouts and fallback resources.

## Upstream review

The feature is registered as a separate module. Changes to existing production
files are limited to registration, optional export-plan plumbing, original-bone
audit masking, per-draw fallback binding, and managed preview modifier handling.
The feature does not install anything into a running game, rewrite captured
workspace profiles, or change existing extraction and texture workflows.

## Contributors

Contributors to this feature:

- [Hajimicat-1](https://github.com/Hajimicat-1): feature specification, workflow design, in-game testing, and validation.
- ChatGPT (OpenAI): AI-assisted implementation, debugging, regression testing, and technical review.
