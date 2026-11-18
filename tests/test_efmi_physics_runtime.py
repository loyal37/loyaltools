"""Offline serializer/INI contract tests. No Blender, game, or GPU is required.

The tiny INI interpreter executes the GENERATED host commands against an
independent publication model; it does not claim to execute shader dynamics.
Actual cs_5_0 compilation is a separate validation step.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import re
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "efmi_physics" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("efmi_physics_runtime_under_test", PATH)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def matrix(x=0, y=0, z=0, turn=False):
    # +Y longitudinal axis, with optional 90 degree turn about Z.
    return ([[0., -1., 0., x], [1., 0., 0., y], [0., 0., 1., z], [0., 0., 0., 1.]] if turn else
            [[1., 0., 0., x], [0., 1., 0., y], [0., 0., 1., z], [0., 0., 0., 1.]])


def fixture():
    # Stable ID 11 was deleted. GPU palette entries are explicit and deliberately
    # not in chain traversal order (10,12 followed by 20).
    plan = {"base_bone_count": 10, "bones": [
        {"global_id": 20, "palette_index": 11, "parent_global_id": 7,
         "rest_matrix": matrix(3), "length": .4},
        {"global_id": 12, "palette_index": 12, "parent_global_id": 10,
         "rest_matrix": matrix(0, 1, turn=True), "length": .6},
        {"global_id": 10, "palette_index": 10, "parent_global_id": 2,
         "rest_matrix": matrix(), "length": 1.}],
        "chains": [{"anchor": 2, "segments": [10, 12]}, {"anchor": 7, "segments": [20]}],
        "meshes": [{"object_name": "cape"}, {"object_name": "ribbon"}]}
    profile = {"adapter": {"kind": "efmi-rolling-buffer-v1", "original_bones": 10,
                           "instances": 2, "implicit_bones": 512},
               "input_contract": {"validated_source": "rolling"},
               "components": [{"id": 0, "offset": 0, "count": 5},
                              {"id": 2, "offset": 5, "count": 5}]}
    return plan, profile


class HostINI:
    """Execute only the expression/branch/pool subset emitted by the generator."""
    def __init__(self, artifact):
        self.variables = {"$mod_enabled": 1, "FRAME_NUMBER": 0, "time": 0.,
                          "x0": 0., "y0": 0., "z0": 0., "w0": 0.,
                          "x1": 0., "y1": 0., "z1": 0., "w1": 0.,
                          "x2": 0., "y2": 0., "z2": 0., "w2": 0.}
        self.pools = {}
        self.sections = {}
        self.handlers = {}
        section = None
        for raw in artifact.ini_text.splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                section = line[1:-1]
                self.sections.setdefault(section, [])
            else:
                self.sections[section].append(line)
        for line in self.sections["Constants"]:
            match = re.fullmatch(r"global (\$\w+) = ([-\d]+)", line)
            self.variables[match[1]] = int(match[2])
        for name, lines in self.sections.items():
            if name.startswith("Pool"):
                values = dict(line.split(" = ", 1) for line in lines)
                self.pools["$"+name] = [float(values.get("pool_variable_default_value", 0))] * int(values["pool_size"])

    def evaluate(self, expression, local):
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        expression = re.sub(r"\$[A-Za-z_\\][A-Za-z0-9_\\]*|\b(?:FRAME_NUMBER|time|[xyzw][012])\b",
                            lambda match: f"get({match[0]!r})", expression)
        def get(name):
            return local[name] if name in local else self.pools.get(name, self.variables.get(name, 0))
        return eval(expression, {"__builtins__": {}}, {"get": get})

    def run(self, name):
        if name in self.handlers:
            self.handlers[name]()
            return
        if name.startswith("CommandList\\EFMIv1\\"):
            return
        local = {}
        stack = []
        active = True
        for line in self.sections[name]:
            if line.startswith("if "):
                condition = active and bool(self.evaluate(line[3:], local))
                stack.append([active, condition])
                active = condition
            elif line.startswith("elif "):
                parent, taken = stack[-1]
                active = parent and not taken and bool(self.evaluate(line[5:], local))
                stack[-1][1] |= active
            elif line == "else":
                parent, taken = stack[-1]
                active = parent and not taken
                stack[-1][1] = True
            elif line == "endif":
                active = stack.pop()[0]
            elif not active:
                continue
            elif line.startswith("run = "):
                self.run(line[6:])
            elif line.startswith("local "):
                target, expression = line[6:].split(" = ", 1)
                local[target] = self.evaluate(expression, local)
            elif re.match(r"(?:\$|[xyzw][012] = )", line):
                target, expression = line.split(" = ", 1)
                value = self.evaluate(expression, local)
                match = re.fullmatch(r"(\$\w+)\[(.*)\]", target)
                if match:
                    self.pools[match[1]][int(self.evaluate(match[2], local))] = value
                elif target in local:
                    local[target] = value
                else:
                    self.variables[target] = value


class Publication:
    """Independent GPU pending/current/history validity model, per chain+slot."""
    def __init__(self):
        self.states = {}
        self.integrations = {}

    def prepare(self, host):
        v = host.variables
        key = (int(v["$EFMIPhysics_slot"]), int(v["$EFMIPhysics_chain"]))
        frame, generation = int(v["FRAME_NUMBER"]) & 1048575, int(v["$EFMIPhysics_generation"])
        old = self.states.get(key)
        if old is None or old["generation"] != generation:
            preserve = old is not None and old["frame"] == frame
            state = dict(generation=generation, frame=frame, last_step=-1,
                         pending=None, current=None, history=None, reset_history=True)
            if preserve:
                state.update(current=old["current"], history=old["history"])
        else:
            state = old
            if state["frame"] != frame:
                history = state["current"]
                if state["pending"] is not None:
                    state["current"] = state["pending"]
                    if history is None or state["reset_history"]:
                        history = state["current"]
                    state["reset_history"] = False
                state.update(frame=frame, history=history)
        if v["$EFMIPhysics_allow_integrate"] and v["$EFMIPhysics_anchor_ready"] and state["last_step"] != frame:
            state.update(pending=(generation, frame, key), last_step=frame)
            self.integrations[(key, frame)] = self.integrations.get((key, frame), 0)+1
        self.states[key] = state


class RuntimeTests(unittest.TestCase):
    def test_multi_chain_multisegment_sparse_id_roundtrip(self):
        plan, profile = fixture()
        before = deepcopy((plan, profile))
        normalized = runtime.normalize_runtime_plan(plan, profile)
        self.assertEqual((plan, profile), before)
        self.assertEqual([bone["global_id"] for bone in normalized["bones"]], [10, 12, 20])
        self.assertEqual([bone["palette_index"] for bone in normalized["bones"]], [10, 12, 11])
        self.assertEqual([chain["first_segment"] for chain in normalized["chains"]], [0, 2])
        rows = runtime.profile_rows(normalized)
        self.assertEqual(len(rows), 4+2*2+12*3)
        self.assertEqual(rows[1], [2, 4, 8, 12])
        self.assertEqual(rows[8], [10, 10, -1, 1.])
        self.assertEqual(rows[20], [12, 12, 0, .6])
        self.assertEqual(rows[32], [11, 20, -1, .4])
        self.assertEqual(rows[15], [0., -9.8, 0., .08])
        artifact = runtime.build_physics_runtime(plan, profile)
        self.assertEqual(artifact.metadata["extended_bones"], 13)
        self.assertEqual(artifact.metadata["global_to_palette"], {10: 10, 12: 12, 20: 11})
        saved = json.loads(artifact.files["res/EFMIPhysics/profile.json"])
        self.assertEqual(saved, normalized)

    def test_parameter_overrides_and_edited_restframe_are_preserved(self):
        plan, profile = fixture()
        plan["chains"][0]["params"] = {"gravity": [1, -2, 3], "damping": .25, "cone_angle": 90}
        plan["bones"][1]["params"] = {"stiffness": .9}
        normalized = runtime.normalize_runtime_plan(plan, profile)
        child = normalized["bones"][1]
        self.assertEqual(child["rest_matrix"], matrix(0, 1, turn=True))
        self.assertEqual(child["params"]["gravity"], [1, -2, 3])
        self.assertEqual(child["params"]["damping"], .25)
        self.assertEqual(child["params"]["stiffness"], .9)
        self.assertEqual(child["length"], .6)

    def test_authoring_ids_above_r16_map_to_explicit_dense_palette(self):
        plan, profile = fixture()
        ids = {10: 65536, 12: 16777217, 20: runtime.MAX_AUTHORING_ID}
        for bone in plan["bones"]:
            bone["global_id"] = ids[bone["global_id"]]
            bone["parent_global_id"] = ids.get(bone["parent_global_id"], bone["parent_global_id"])
        for chain in plan["chains"]:
            chain["segments"] = [ids[gid] for gid in chain["segments"]]
        normalized = runtime.normalize_runtime_plan(plan, profile)
        artifact = runtime.build_physics_runtime(plan, profile)
        self.assertEqual(artifact.metadata["global_to_palette"], {65536: 10, 16777217: 12, 2147483647: 11})
        self.assertTrue(all(10 <= bone["palette_index"] <= 65535 for bone in normalized["bones"]))
        self.assertEqual([bone["parent_global_id"] for bone in normalized["bones"]], [2, 65536, 7])
        saved = json.loads(artifact.files["res/EFMIPhysics/profile.json"])
        self.assertEqual([bone["global_id"] for bone in saved["bones"]], [65536, 16777217, 2147483647])
        rows = runtime.profile_rows(normalized)
        self.assertEqual([rows[8 + 12*index][0] for index in range(3)], [10, 12, 11])
        self.assertEqual(rows[20][2], 0)  # Parent is a segment ordinal, never an authoring ID.
        invalid = deepcopy(plan)
        invalid["bones"][0]["global_id"] = runtime.MAX_AUTHORING_ID + 1
        with self.assertRaisesRegex(ValueError, "custom global ID"):
            runtime.build_physics_runtime(invalid, profile)
        invalid = deepcopy(plan)
        invalid["bones"][1]["parent_global_id"] = runtime.MAX_AUTHORING_ID + 1
        with self.assertRaisesRegex(ValueError, "segment parent ID"):
            runtime.build_physics_runtime(invalid, profile)

    def test_invalid_profiles_fail_before_files_are_returned(self):
        cases = []
        plan, profile = fixture()
        cases.append((plan, dict(profile, input_contract={"validated_source": "cb"}), "posing source"))
        cases.append((plan, dict(profile, input_contract={}), "posing source"))
        cases.append((plan, dict(profile, adapter=dict(profile["adapter"], instances=33)), "instance count"))
        for field, value, message in (("global_id", 9, "global ID"), ("length", 0., "length"),
                                      ("palette_index", 10, "Duplicate palette")):
            mutated = deepcopy(plan)
            mutated["bones"][0][field] = value
            cases.append((mutated, profile, message))
        mutated = deepcopy(plan)
        mutated["bones"][0]["rest_matrix"][0][0] = 2
        cases.append((mutated, profile, "orthogonal axes"))
        mutated = deepcopy(plan)
        mutated["bones"][1]["rest_matrix"][0][3] = 8
        cases.append((mutated, profile, "connect"))
        mutated = deepcopy(plan)
        mutated["bones"][0]["params"] = {"gravity": [0., float("nan"), 0.]}
        cases.append((mutated, profile, "finite"))
        mutated = deepcopy(plan)
        mutated["chains"][1]["anchor"] = 10
        cases.append((mutated, profile, "native anchor"))
        for p, f, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                runtime.build_physics_runtime(p, f)
        with self.assertRaisesRegex(ValueError, "namespace"):
            runtime.build_physics_runtime(plan, profile, namespace="Unsafe\n[Present]")

    def test_capacity_bounds_and_empty_render_components(self):
        plan, profile = fixture()
        profile["components"].append({"id": 3, "offset": 10, "count": 0})
        artifact = runtime.build_physics_runtime(plan, profile)
        self.assertIn("CommandList_Draw_Component3", artifact.metadata["callback_wrappers"])
        # A native B=65535 plus one extra index is valid through R16 index65535.
        edge_plan = {"base_bone_count": 65535,
                     "bones": [{"global_id": 65535, "rest_matrix": matrix(), "length": 1}],
                     "chains": [{"anchor": 2, "segments": [65535]}]}
        edge_profile = {"adapter": {"original_bones": 65535, "instances": 32, "implicit_bones": 512},
                        "input_contract": {"validated_source": "rolling"},
                        "components": [{"id": 0, "offset": 0, "count": 65535}]}
        artifact = runtime.build_physics_runtime(edge_plan, edge_profile)
        self.assertEqual(artifact.metadata["extended_bones"], 65536)
        plan, profile = fixture()
        plan["bones"] = [{"global_id": 10+i, "rest_matrix": matrix(y=i), "length": 1} for i in range(65)]
        plan["chains"] = [{"anchor": 2, "segments": list(range(10, 75))}]
        with self.assertRaisesRegex(ValueError, "1..64"):
            runtime.build_physics_runtime(plan, profile)

    def test_generated_wrapper_and_temporary_resource_lifetime(self):
        artifact = runtime.build_physics_runtime(*fixture())
        ini = artifact.ini_text
        for draw, wrapper in artifact.metadata["callback_wrappers"].items():
            self.assertIn(f"[{wrapper}]\nrun = CommandListEFMIPhysics_Begin\nrun = {draw}\nrun = CommandListEFMIPhysics_End", ini)
        self.assertNotRegex(ini, r"(?m)^\$\\EFMIv1\\bones_count\s*=")
        self.assertNotIn("o0 =", ini)
        self.assertNotIn("backbuffer", ini.split("Never retain a render target or backbuffer.")[0])
        for name, binding in (("CS_CB0", "cs-cb0"), ("CS_T0", "cs-t0"), ("CS_T1", "cs-t1"),
                              ("CS_U0", "cs-u0"), ("CS_U1", "cs-u1"), ("CS_U2", "cs-u2"), ("VS_T0", "vs-t0")):
            self.assertIn(f"ResourceEFMIPhysics_Save_{name} = ref {binding}", ini)
            modifier = "" if name == "CS_CB0" else "no_view_cache "
            self.assertIn(f"{binding} = {modifier}ref ResourceEFMIPhysics_Save_{name}", ini)
            self.assertIn(f"ResourceEFMIPhysics_Save_{name} = ref iniparams\n    ResourceEFMIPhysics_Save_{name} = null", ini)
        self.assertIn("y2 = $\\EFMIv1\\cfg_ms_implicit_bones_count", ini)
        self.assertIn("cfg_ms_implicit_bones_count > 512", ini)
        self.assertIn("array = 16", ini)  # two chains * two slots * four metadata rows
        patch = ini.split("[CustomShaderEFMIPhysics_PatchConfig]", 1)[1].split("[CommandListEFMIPhysics_End]", 1)[0]
        self.assertIn("cs-cb0 = ref ResourceEFMIPhysics_Save_CS_CB0", patch)

    def test_generated_cpu_ready_matches_independent_gpu_publication(self):
        artifact = runtime.build_physics_runtime(*fixture())
        orders = sorted(set(permutations((0, 0, 2, 2))))
        for order in orders:
            with self.subTest(order=order):
                host, gpu = HostINI(artifact), Publication()
                v = host.variables
                v.update({"$\\EFMIv1\\merged_skeleton_connected": 1,
                          "$\\EFMIv1\\bones_data_source": 1, "$\\EFMIv1\\instance_config_vs_cb_slot": 2,
                          "$\\EFMIv1\\bones_count": 10, "$\\EFMIv1\\instance_count": 2,
                          "$\\EFMIv1\\cfg_ms_implicit_bones_count": 512,
                          "$\\EFMIv1\\merged_skeleton_instance_id": 0})
                host.pools["$Pool_MergedSkeleton_Instance_UpdateFrame"] = [-1]*6
                host.pools["$Pool_MergedSkeleton_Instance_LodLevel"] = [0]*6
                host.handlers["CustomShaderEFMIPhysics_Prepare"] = lambda: gpu.prepare(host)
                host.handlers["CustomShaderEFMIPhysics_PatchConfig"] = lambda: None
                trace = []
                def observe():
                    state = tuple((gpu.states[(0, ci)]["current"], gpu.states[(0, ci)]["history"]) for ci in range(2))
                    readiness = tuple(bool(v[f"$EFMIPhysics_chain_ready_{ci}"]) for ci in range(2))
                    self.assertEqual(readiness, tuple(pair[0] is not None for pair in state))
                    trace.append((state, readiness))
                for original in artifact.metadata["callback_wrappers"]:
                    host.handlers[original] = observe
                # Distinct anchor components become fresh at different callbacks.
                # Frame 12 changes LoD only when each anchor callback arrives.
                for frame in range(10, 15):
                    v.update(FRAME_NUMBER=frame, time=frame/60.)
                    trace.clear()
                    for component in order:
                        host.pools["$Pool_MergedSkeleton_Instance_UpdateFrame"][component*2] = frame
                        host.pools["$Pool_MergedSkeleton_Instance_LodLevel"][component*2] = int(frame >= 12)
                        v["$\\EFMIv1\\component_id"] = component
                        host.run(artifact.metadata["callback_wrappers"][f"CommandList_Draw_Component{component}"])
                    self.assertEqual(len(set(trace)), 1, "draws must use frozen chain poses/readiness all frame")
                self.assertTrue(all(n == 1 for n in gpu.integrations.values()), "each chain integrates at most once per slot/frame")
                self.assertEqual(len(gpu.integrations), 10)

    def test_unsupported_runtime_source_does_not_bind_or_dispatch(self):
        artifact = runtime.build_physics_runtime(*fixture())
        host = HostINI(artifact)
        host.variables.update({"$\\EFMIv1\\merged_skeleton_connected": 1,
                               "$\\EFMIv1\\bones_data_source": 2,
                               "$\\EFMIv1\\instance_config_vs_cb_slot": 2})
        host.run("CommandListEFMIPhysics_Begin")
        self.assertEqual(host.variables["$EFMIPhysics_bound"], 0)
        self.assertEqual(host.variables["$EFMIPhysics_unsupported_reason"], 1)


if __name__ == "__main__":
    unittest.main()
