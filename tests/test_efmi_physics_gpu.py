"""Actual SM5 dispatch/readback on a windowless WARP software D3D11 device.

Run on Windows: python -m unittest discover -s tests -p test_efmi_physics_gpu.py -v
This never accesses the game, a display, DXGI swap chains, or injection tools.
"""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_runtime():
    name = "efmi_physics_gpu_runtime"
    spec = importlib.util.spec_from_file_location(name, ROOT / "efmi_physics/runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture():
    bones, chains = [], []
    for ci, (anchor, root, lengths, ids) in enumerate([
        (0, [1., 2., 3.], [1., .5], [65536, 2147483647]),
        (2, [-2., 1., 0.], [.6, .8], [9, 12]),
    ]):
        chains.append({"anchor": anchor, "segments": ids,
                       "params": {"gravity": [2., 0., 0.], "stiffness": 0., "damping": 0., "cone_angle": 90.}})
        head = np.array(root)
        parent = anchor
        for j, (gid, length) in enumerate(zip(ids, lengths)):
            rest = np.eye(4)
            rest[:3, 3] = head
            bones.append({"global_id": gid, "palette_index": 4 + 2*ci + j,
                          "parent_global_id": parent, "rest_matrix": rest.tolist(),
                          "inverse_bind": np.linalg.inv(rest).tolist(), "length": length})
            head = head + [0, length, 0]
            parent = gid
    plan = {"base_bone_count": 4, "bones": bones, "chains": chains}
    profile = {"adapter": {"kind": "efmi-rolling-buffer-v1", "original_bones": 4,
                            "instances": 2, "implicit_bones": 2},
               "input_contract": {"validated_source": "rolling"},
               "components": [{"id": 0, "offset": 0, "count": 2}, {"id": 1, "offset": 2, "count": 2}]}
    return plan, profile


@unittest.skipUnless(sys.platform == "win32", "Requires Windows D3D11 WARP and D3DCompiler")
class PhysicsGPUDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from d3d11_warp import Warp
        cls.runtime = load_runtime()
        cls.warp = Warp()
        cls.prepare_shader = cls.warp.shader(ROOT / "efmi_physics/shaders/prepare.hlsl")
        cls.patch_shader = cls.warp.shader(ROOT / "efmi_physics/shaders/config_patch.hlsl")

    @classmethod
    def tearDownClass(cls):
        cls.warp.close()

    def setUp(self):
        self.plan, self.profile = fixture()
        self.normalized = self.runtime.normalize_runtime_plan(self.plan, self.profile)
        self.B, self.K, self.N, self.I, self.Icap, self.C = 4, 4, 2, 1, 2, 2
        self.E = self.B + self.K
        rows = np.array(self.runtime.profile_rows(self.normalized), dtype=np.float32)
        self.profile_buffer = self.warp.buffer(rows, bind=8)
        self.profile_srv = self.warp.view(self.profile_buffer)
        self.source = np.zeros(((self.I + 2*self.B*self.N)*3, 4), dtype=np.float32)
        self.source[:3] = np.eye(4, dtype=np.float32)[:3]
        self.anchors = {}
        for slot in range(self.N):
            for history in range(2):
                for bone in range(self.B):
                    matrix = np.eye(4, dtype=np.float32)[:3]
                    matrix[:, 3] = [3 + slot*20 + bone*5, 4 + slot*2 + bone - history*.5, slot]
                    self.source[self.source_at(slot, bone, history):self.source_at(slot, bone, history)+3] = matrix
                    if history == 0:
                        self.anchors[slot, bone] = matrix.copy()
        self.source_buffer = self.warp.buffer(self.source, bind=8)
        self.source_srv = self.warp.view(self.source_buffer)
        self.output_shape = ((self.Icap + 2*self.E*self.N)*3, 4)
        self.output_buffer = self.warp.buffer(np.full(self.output_shape, 12345., dtype=np.float32))
        self.output_uav = self.warp.view(self.output_buffer, uav=True)
        self.state_shape = (self.N*self.K*24, 4)
        self.state_buffer = self.warp.buffer(np.zeros(self.state_shape, dtype=np.float32))
        self.state_uav = self.warp.view(self.state_buffer, uav=True)
        self.control_shape = (self.N*self.C*4, 4)
        self.control_buffer = self.warp.buffer(np.zeros(self.control_shape, dtype=np.float32))
        self.control_uav = self.warp.view(self.control_buffer, uav=True)
        self.params = np.zeros((3, 4), dtype=np.float32)
        self.params_texture = self.warp.texture(self.params)
        self.params_srv = self.warp.view(self.params_texture)

    def source_at(self, slot, bone, history=0):
        return (self.I + 2*self.B*slot + self.B*history + bone)*3

    def output_at(self, slot, bone, history=0):
        return (self.I + 2*self.E*slot + self.E*history + bone)*3

    def dispatch(self, frame, slot, chain, *, fresh=True, integrate=True, simulate=False, generation=0, epoch=0):
        self.params[:] = [[frame, slot, fresh, integrate], [1/60, frame/60, generation, epoch],
                          [simulate, self.I, chain, 0]]
        self.warp.update(self.params_texture, self.params)
        self.warp.dispatch(self.prepare_shader,
                           {0: self.source_srv, 1: self.profile_srv, 120: self.params_srv},
                           [self.output_uav, self.state_uav, self.control_uav],
                           math.ceil((self.Icap*3 + 2*self.B*3)/64))

    def read_output(self):
        return self.warp.read(self.output_buffer, self.output_shape)

    def read_state(self):
        return self.warp.read(self.state_buffer, self.state_shape)

    def read_control(self):
        return self.warp.read(self.control_buffer, self.control_shape)

    def prepare_all(self, frame, **kwargs):
        for slot in range(self.N):
            for chain in range(self.C):
                self.dispatch(frame, slot, chain, **kwargs)

    def assert_base_and_guards(self, output):
        np.testing.assert_array_equal(output[:self.I*3], self.source[:self.I*3])
        for slot in range(self.N):
            for history in range(2):
                start = self.output_at(slot, 0, history)
                source = self.source_at(slot, 0, history)
                np.testing.assert_array_equal(output[start:start+self.B*3], self.source[source:source+self.B*3])
        # Allocation uses Icap, addresses use Iactual; the unused tail is a guard.
        np.testing.assert_array_equal(output[(self.I+2*self.E*self.N)*3:], 12345.)

    def test_dispatch_two_chains_two_segments_and_two_slots(self):
        # Stable authoring IDs are not GPU indices. Even a float32-rounded
        # diagnostic ID in the existing profile cannot change dense skinning.
        self.assertGreater(max(bone["global_id"] for bone in self.plan["bones"]), 65535)
        self.assertEqual([bone["palette_index"] for bone in self.normalized["bones"]], [4, 5, 6, 7])
        self.assertTrue(all(bone["palette_index"] <= 65535 for bone in self.normalized["bones"]))
        self.prepare_all(1)
        first = self.read_output()
        self.assert_base_and_guards(first)
        for slot in range(self.N):
            for bone in range(self.B, self.E):
                for history in range(2):
                    start = self.output_at(slot, bone, history)
                    np.testing.assert_array_equal(first[start:start+3], np.eye(4)[:3])
        self.prepare_all(2)
        second = self.read_output()
        self.assertTrue(np.isfinite(second).all())
        self.assert_base_and_guards(second)
        control = self.read_control()
        for slot in range(self.N):
            for chain in range(self.C):
                at = (slot*self.C + chain)*4
                # Numeric encoding avoids float denormals losing small frame
                # numbers through SM5 branch joins on a real D3D11 device.
                np.testing.assert_array_equal(control[at, 1:3], [2, 2])
                self.assertEqual(control[at+1, 3], 2)
        for slot in range(self.N):
            for ci, anchor in enumerate((0, 2)):
                for segment in range(2):
                    start = self.output_at(slot, self.B + 2*ci + segment)
                    np.testing.assert_allclose(second[start:start+3], self.anchors[slot, anchor], atol=2e-5)
                    np.testing.assert_allclose(second[start+self.E*3:start+self.E*3+3], second[start:start+3], atol=2e-5)
        state = self.read_state()
        for slot in range(self.N):
            for bi, bone in enumerate(self.plan["bones"]):
                at = (slot*self.K + bi)*24
                self.assertAlmostEqual(np.linalg.norm(state[at, :3] - state[at+2, :3]), bone["length"], places=4)
                if bi % 2:
                    np.testing.assert_allclose(state[at+2, :3], state[at-24, :3], atol=2e-5)
                else:
                    bind_head = np.array(bone["rest_matrix"])[:, 3]
                    expected = self.anchors[slot, (0, 2)[bi//2]] @ bind_head
                    np.testing.assert_allclose(state[at+2, :3], expected, atol=2e-5)

    def test_publication_frozen_per_frame_history_and_anchor_freshness(self):
        self.prepare_all(1)
        self.prepare_all(2)
        before = self.read_output()
        state = self.read_state()
        count = self.read_control()[1, 3]
        # A second component callback changes its anchor, but cannot revise
        # an already published frame or integrate the same chain twice.
        start = self.source_at(0, 0)
        self.source[start:start+3, 3] += [.1, 0, 0]
        self.warp.update(self.source_buffer, self.source)
        for chain in (1, 0, 1, 0):
            self.dispatch(2, 0, chain, simulate=True)
        duplicate = self.read_output()
        np.testing.assert_array_equal(duplicate[self.output_at(0, self.B):self.output_at(0, self.E)],
                                      before[self.output_at(0, self.B):self.output_at(0, self.E)])
        self.assertEqual(self.read_control()[1, 3], count)
        # A new frame with a stale anchor may publish last pending, but cannot
        # advance solver state or its integration count.
        self.dispatch(3, 0, 0, fresh=False, integrate=True, simulate=True)
        stale = self.read_state()
        for bi in (0, 1):
            at = bi*24
            np.testing.assert_array_equal(stale[at:at+5], state[at:at+5])
        self.assertEqual(self.read_control()[1, 3], count)
        published = self.read_output()
        for bone in (4, 5):
            current = self.output_at(0, bone)
            previous = self.output_at(0, bone, 1)
            np.testing.assert_allclose(published[previous:previous+3], before[current:current+3], atol=2e-5)
        # Slot 1 is a separate spatial instance and was not dispatched again.
        np.testing.assert_array_equal(published[self.output_at(1, 0):self.output_at(1, self.E, 1)],
                                      before[self.output_at(1, 0):self.output_at(1, self.E, 1)])

    def test_simulated_chain_length_finite_and_late_reset_preserves_draw(self):
        self.prepare_all(1, simulate=True)
        for frame in range(2, 14):
            self.prepare_all(frame, simulate=True)
        output = self.read_output()
        state = self.read_state()
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.isfinite(state).all())
        self.assert_base_and_guards(output)
        for slot in range(self.N):
            for bi, bone in enumerate(self.plan["bones"]):
                at = (slot*self.K + bi)*24
                self.assertAlmostEqual(np.linalg.norm(state[at, :3] - state[at+2, :3]), bone["length"], places=4)
                if bi % 2:
                    np.testing.assert_allclose(state[at+2, :3], state[at-24, :3], atol=3e-5)
        self.assertGreater(abs(state[0, 0] - state[2, 0]), 1e-4, "Actual gravity must bend the chain")
        # Current pose is frame 12, pending is frame 13. A repeated draw in
        # frame 13 must not promote its different pending pose a second time.
        self.dispatch(13, 0, 0, simulate=True)
        duplicate = self.read_output()
        for history in (0, 1):
            start = self.output_at(0, 4, history)
            np.testing.assert_array_equal(duplicate[start:start+6], output[start:start+6])
        # A late LoD epoch/generation changes next pending, keeping consumed C/H.
        self.dispatch(13, 0, 0, simulate=True, generation=1, epoch=1)
        reset = self.read_output()
        for history in (0, 1):
            start = self.output_at(0, 4, history)
            np.testing.assert_array_equal(reset[start:start+6], output[start:start+6])

    def test_config_patch_changes_only_dynamic_offsets(self):
        original = np.arange(64, dtype=np.uint32).reshape(16, 4)
        original[5] = [30, 60, 1, 2]
        cb = self.warp.buffer(original, bind=4, structured=False)
        output = self.warp.buffer(np.zeros_like(original), bind=0x80, structured=False)
        view = self.warp.view(output, uav=True, typed_count=16)
        self.params[:] = [[2, 1, 1, 1], [1/60, 2/60, 0, 0], [0, self.I, 0, 0]]
        self.warp.update(self.params_texture, self.params)
        self.warp.dispatch(self.patch_shader, {1: self.profile_srv, 120: self.params_srv}, [view], constant_buffer=cb)
        actual = self.warp.read(output, (16, 4), dtype=np.uint32)
        expected = original.copy()
        base = (self.I + 2*self.E)*3
        expected[5, :2] = [base-3, base+self.E*3-3]
        np.testing.assert_array_equal(actual, expected)
        original[5, :2] = [0, 2]  # Static/implicit bone offsets must be retained.
        self.warp.update(cb, original)
        self.warp.dispatch(self.patch_shader, {1: self.profile_srv, 120: self.params_srv}, [view], constant_buffer=cb)
        np.testing.assert_array_equal(self.warp.read(output, (16, 4), dtype=np.uint32), original)


if __name__ == "__main__":
    unittest.main()
