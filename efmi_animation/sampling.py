"""Sample authored EFMI skeleton clips without modifying the rig or its keys."""

from collections import defaultdict
import math

import bpy
from mathutils import Matrix

from . import rig


SAMPLES_PER_FRAME = 8
MAX_BONE_SAMPLES = 131072
MATRIX_EPSILON = 1e-5


def _require(condition, message):
    if not condition:
        raise ValueError("EFMI animation export: " + message)


def _active_animation_groups(obj):
    names = {group.index: group.name for group in obj.vertex_groups
             if group.name.startswith(rig.GROUP_PREFIX)}
    if not names:
        return set()
    return {names[weight.group] for vertex in obj.data.vertices for weight in vertex.groups
            if weight.group in names and weight.weight > 0.0}


def validate_animation_support(export_objects, profile, *, enabled):
    """Fail before writing buffers when reserved groups cannot be exported."""
    if enabled and profile is not None:
        return
    for obj in export_objects:
        if obj.type == 'MESH' and _active_animation_groups(obj):
            reason = "Use EFMI Merged Skeleton export." if profile is None else "Enable EFMI Bone Animation."
            _require(False, obj.name + " has weighted animation vertex groups. " + reason)


def _validate_linked_physics_dependencies(scene, export_objects):
    physics = getattr(scene, 'loyaltools_efmi_physics', None)
    if physics is None or not physics.objects:
        return
    from ..efmi_physics.rig import is_managed_export_copy

    for entry in physics.objects:
        source = entry.object
        if source is None:
            continue
        chain = next((item for item in entry.chains if getattr(item, 'animation_anchor_group', '')), None)
        if chain is not None and any(obj == source or is_managed_export_copy(entry, obj) for obj in export_objects):
            _require(False, source.name + '/' + chain.name + " uses an animation bone attachment. "
                     "Enable EFMI Bone Animation or select a native bone attachment.")


def _related_meshes(settings, export_objects, required):
    targets = defaultdict(list)
    entries = [entry for entry in settings.objects if entry.object is not None]
    required_entries = [entry for entry in entries if entry.object.name in required]
    for obj in export_objects:
        if obj.type != 'MESH':
            continue
        active = _active_animation_groups(obj)
        candidates = entries if active else required_entries
        owners = [entry for entry in candidates if entry.object == obj or rig.is_managed_export_copy(entry, obj)]
        _require(len(owners) < 2, obj.name + " has duplicate animation registrations.")
        if not owners:
            _require(not active, obj.name + " has weights on unregistered animation vertex groups: "
                     + ', '.join(sorted(active)))
            continue
        targets[owners[0].object.as_pointer()].append((obj, active))
    return [(entry, targets[entry.object.as_pointer()]) for entry in entries
            if entry.object.as_pointer() in targets]


def _matrix_rows(matrix):
    return [list(row) for row in matrix]


def _matrices_match(first, second):
    return all(abs(a - b) <= MATRIX_EPSILON * max(1.0, abs(a), abs(b))
               for first_row, second_row in zip(first, second)
               for a, b in zip(first_row, second_row))


def _check_matrix(matrix, label):
    _require(all(math.isfinite(value) for row in matrix for value in row), label + " has a non-finite transform.")


def _inverse(matrix, label):
    _check_matrix(matrix, label)
    try:
        inverse = matrix.inverted()
    except ValueError as error:
        raise ValueError("EFMI animation export: " + label + " has a singular transform. Check its scale.") from error
    _check_matrix(inverse, "Inverse transform of " + label)
    return inverse


def _decompose(matrix, previous, label):
    _check_matrix(matrix, label)
    translation, rotation, scale = matrix.decompose()
    _require(all(abs(axis) >= 1e-8 for axis in scale), label + " has a zero scale axis.")
    rotation.normalize()
    _require(_matrices_match(matrix, Matrix.LocRotScale(translation, rotation, scale)),
             label + " has shear. Check bone constraints and inherited scale.")
    if previous is not None:
        _require(all(a * b > 0 for a, b in zip(previous['scale'], scale)),
                 label + " has a scale sign change since the previous sample.")
        if sum(a * b for a, b in zip(previous['rotation'], (rotation.x, rotation.y, rotation.z, rotation.w))) < 0:
            rotation.negate()
    return {'translation': list(translation), 'rotation': [rotation.x, rotation.y, rotation.z, rotation.w],
            'scale': list(scale)}


def _ordered_tree(armature, root):
    records = {record.uid: record for record in root.bones}
    _require(len(records) == len(root.bones) and all(records), root.name + " has invalid or duplicate bone IDs.")
    children = defaultdict(list)
    for record in root.bones:
        _require(not record.parent_uid or record.parent_uid in records, root.name + " has a bone with a missing parent.")
        children[record.parent_uid].append(record)
    _require(len(children['']) == 1, root.name + " must contain exactly one root bone.")
    order = []
    resolved = {}
    stack = list(reversed(children['']))
    while stack:
        record = stack.pop()
        bone = rig.find_bone(armature, record)
        _require(bone is not None and bone.length > 1e-6,
                 root.name + " has a missing or zero-length animation bone: " + record.group_name)
        _require(bone.parent == resolved.get(record.parent_uid),
                 root.name + " has a bone whose Blender parent does not match its registration: " + bone.name)
        resolved[record.uid] = bone
        order.append((record, bone))
        stack.extend(reversed(children[record.uid]))
    _require(len(order) == len(records), root.name + " has a cycle or disconnected branch in its bone hierarchy.")
    return order


def _sample(scene, tracks, plan, mirror):
    by_tick = defaultdict(list)
    for track in tracks:
        chain, _, _ = track
        first = chain['frame_start'] * SAMPLES_PER_FRAME
        for tick in range(first, first + chain['sample_count']):
            by_tick[tick].append(track)
    starting_matrices = {}
    # Keep the sampling graph inactive. Evaluated animation then stays in the
    # graph and is not copied back over the original pose or other preview values.
    sample_scene = scene.copy()
    try:
        view_layer = sample_scene.view_layers[bpy.context.view_layer.name]
        for tick in sorted(by_tick):
            frame, substep = divmod(tick, SAMPLES_PER_FRAME)
            subframe = substep / SAMPLES_PER_FRAME
            sample_scene.frame_set(frame, subframe=subframe)
            _require(sample_scene.frame_current + sample_scene.frame_subframe == frame + subframe,
                     f"Could not sample frame {frame + subframe:g}. Check Blender's frame range.")
            depsgraph = view_layer.depsgraph
            evaluated_rigs = {}
            for chain, armature, indices in by_tick[tick]:
                if armature not in evaluated_rigs:
                    evaluated_rigs[armature] = armature.evaluated_get(depsgraph)
                evaluated = evaluated_rigs[armature]
                current_world = mirror @ evaluated.matrix_world @ mirror
                _require(_matrices_match(chain['root_transform'], current_world),
                         armature.name + " has an animated object transform. Keyframe location or rotation on its root bone.")
                inverse_poses = {}
                for index in indices:
                    bone = plan['bones'][index]
                    label = f"{armature.name}/{bone['bone_name']} at frame {frame + subframe:g}"
                    pose = evaluated.pose.bones.get(bone['bone_name'])
                    _require(pose is not None, "Blender could not evaluate " + label + ".")
                    local = pose.matrix
                    parent = bone['parent_index']
                    if parent != -1:
                        if parent not in inverse_poses:
                            parent_name = plan['bones'][parent]['bone_name']
                            inverse_poses[parent] = _inverse(evaluated.pose.bones[parent_name].matrix, parent_name)
                        local = inverse_poses[parent] @ local
                    local = mirror @ local @ mirror
                    frames = bone['frames']
                    if not frames:
                        starting_matrices[index] = local.copy()
                    if len(frames) + 1 == chain['sample_count']:
                        chain['seamless_loop'] &= _matrices_match(starting_matrices.pop(index), local)
                    frames.append(_decompose(local, frames[-1] if frames else None, label))
    finally:
        bpy.data.scenes.remove(sample_scene)


def prepare_animation_plan(scene, export_objects, profile, *, mirror_x=False, palette_start=None,
                           required_animation_groups=None):
    """Collect and sample directly from verified original Blender armatures.

    Root samples are in converted armature space. Descendants are parent-local,
    so interpolating them preserves branch motion. For bone i, runtime uses
    ``anchor_skin @ root_transform @ posed_chain_i @ inverse_bind_i``.
    """
    export_objects = list(export_objects)
    required = {source: set(groups) for source, groups in (required_animation_groups or {}).items() if groups}
    settings = getattr(scene, 'loyaltools_efmi_animation', None)
    enabled = bool(settings and settings.enabled)
    if not enabled:
        _validate_linked_physics_dependencies(scene, export_objects)
    _require(not required or (enabled and profile is not None),
             "A physics chain uses an animation bone attachment. Enable EFMI Bone Animation and use Merged Skeleton export.")
    validate_animation_support(export_objects, profile, enabled=enabled)
    if not enabled or profile is None:
        return None
    bindings = _related_meshes(settings, export_objects, required)
    if not bindings:
        _require(not required, "The requested physics chain attachments have no matching mesh in this export.")
        return None

    base = int(profile['bones_count'])
    start = base if palette_start is None else int(palette_start)
    _require(0 < base <= start <= 65536, "The extended bone slots exceed the R16_UINT range.")
    fps = float(scene.render.fps) / float(scene.render.fps_base)
    _require(math.isfinite(fps) and fps > 0.0, "Blender's frame rate must be finite and positive.")
    mirror = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0)) if mirror_x else Matrix.Identity(4)
    plan = {'base_bone_count': base, 'bones': [], 'chains': [], 'meshes': [],
            'group_to_palette': {}, 'global_to_palette': {}, 'export_space_transform': _matrix_rows(mirror)}
    tracks = []
    required_sources = set(required)
    total_samples = 0
    for entry, targets in bindings:
        source, armature = entry.object, entry.rig
        requested = required.get(source.name, set())
        required_sources.discard(source.name)
        active = set().union(*(names for _, names in targets))
        rig._verify_entry(entry)
        _require(armature.mode != 'EDIT' and armature.data.pose_position == 'POSE',
                 "Exit Edit Mode and set " + armature.name + " to Pose Position.")
        _require(armature.name in bpy.context.view_layer.objects,
                 armature.name + " is outside the current view layer and cannot be sampled.")
        model_from_rig = mirror @ armature.matrix_world @ mirror
        _check_matrix(model_from_rig, armature.name)
        declared = {record.group_name: int(record.bone_id) for root in entry.roots for record in root.bones}
        unknown = active - declared.keys()
        _require(not unknown, source.name + " has unbound animation vertex groups: " + ', '.join(sorted(unknown)))
        missing = requested - declared.keys()
        _require(not missing, source.name + " is missing animation bone attachments used by physics chains: "
                 + ', '.join(sorted(missing)))

        for root in entry.roots:
            if not (active | requested).intersection(record.group_name for record in root.bones):
                continue
            component = rig.require_follow_component(source, root, profile)
            frame_start, frame_end = int(root.frame_start), int(root.frame_end)
            speed = float(root.playback_speed)
            _require(frame_start <= frame_end, root.name + " must start at or before its end frame.")
            _require(math.isfinite(speed) and speed > 0.0, root.name + " must have a finite, positive playback speed.")
            sample_count = (frame_end - frame_start) * SAMPLES_PER_FRAME + 1
            order = _ordered_tree(armature, root)
            total_samples += sample_count * len(order)
            _require(total_samples <= MAX_BONE_SAMPLES,
                     f"Animation exceeds {MAX_BONE_SAMPLES} bone samples. Shorten the frame range or export fewer bones.")
            _require(start + len(plan['bones']) + len(order) <= 65536, "The extended palette exceeds R16_UINT capacity.")
            chain_index = len(plan['chains'])
            chain = {'chain_id': root.uid, 'name': root.name, 'anchor': int(root.follow_group_id),
                     'component_id': int(component['component_id']),
                     'frame_start': frame_start, 'frame_end': frame_end, 'speed': speed, 'fps': fps,
                     'samples_per_frame': SAMPLES_PER_FRAME, 'sample_count': sample_count,
                     'seamless_loop': True, 'root_transform': _matrix_rows(model_from_rig),
                     'segments': [], 'source_object_name': source.name, 'rig_name': armature.name}
            plan['chains'].append(chain)
            ordinals = {}
            for record, rest_bone in order:
                global_id, group_name = int(record.bone_id), record.group_name
                _require(global_id < 0 and str(global_id) not in plan['global_to_palette'],
                         root.name + " has invalid or duplicate animation bone IDs.")
                _require(group_name.startswith(rig.GROUP_PREFIX) and group_name not in plan['group_to_palette'],
                         root.name + " has an invalid or duplicate animation vertex group: " + group_name)
                _require(source.vertex_groups.get(group_name) is not None,
                         source.name + " is missing animation vertex group " + group_name + ".")
                rest = model_from_rig @ mirror @ rest_bone.matrix_local @ mirror
                inverse_bind = _inverse(rest, "Bind matrix of " + armature.name + '/' + rest_bone.name)
                ordinal = len(plan['bones'])
                palette = start + ordinal
                plan['bones'].append({'global_id': global_id, 'group_name': group_name, 'bone_name': rest_bone.name,
                                      'palette_index': palette, 'chain_index': chain_index, 'frames': [],
                                      'parent_index': ordinals.get(record.parent_uid, -1),
                                      'inverse_bind': _matrix_rows(inverse_bind)})
                ordinals[record.uid] = ordinal
                chain['segments'].append(global_id)
                plan['global_to_palette'][str(global_id)] = palette
                plan['group_to_palette'][group_name] = palette
            tracks.append((chain, armature, list(ordinals.values())))

        for target, target_active in targets:
            if target_active:
                plan['meshes'].append({'object_name': target.name, 'source_object_name': source.name,
                                       'rig_name': armature.name, 'active_group_names': sorted(target_active),
                                       'vg_mapping': {name: declared[name] for name in sorted(target_active)}})

    _require(not required_sources, "Meshes with physics chain attachments missing from this export: "
             + ', '.join(sorted(required_sources)))
    if not tracks:
        return None
    _sample(scene, tracks, plan, mirror)
    plan.update(bone_count=len(plan['bones']), extended_bone_count=start + len(plan['bones']))
    return plan
