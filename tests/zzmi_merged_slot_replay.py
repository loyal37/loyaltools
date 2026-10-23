# -*- coding: utf-8 -*-
"""Replay of the same-frame skinning's slot bookkeeping against a model of XXMI's pools.

Run with ``py tests/zzmi_merged_slot_replay.py`` (no Blender, no GPU).

XXMI's ``fifo`` pools hand out slots in turn: a new key takes the next slot even when that
slot's key is still in use and others are free, and the slot keeps what its previous key left
(an expired slot is reset; a reset resource element stays null for good).  The model follows
``CustomResourcePool::GetElementIndex`` / ``AssignSlot`` / ``ExpireElements`` of XXMI-Libs.

One copy of the character stays on screen while up to seven others come and go (each with new
stream-output buffers), so the pools wrap many times.  Every frame each copy imports its
components' bones at their skinning draws, then draws them in several passes: every part
records where it is, the merged part is skinned once (``CommandListZZMS_SkinNow``) and drawn.
Whatever happens to the slots, a copy must never:

- be drawn from an output that is not its result of this frame (an inherited "skinned" mark,
  a null buffer, another copy's output, or an output that was not the one skinned);
- get as its previous output (vb3) another buffer's result (it is repaired instead);
- see in a store slot a header, record or owner entry that another buffer left there.

Earlier bookkeeping is replayed too and must be caught: v1.7.5 test1 (marks kept by buffer
identity, outputs that expire) and test2 (outputs looked up by buffer identity at each use).
"""

from __future__ import annotations

import random
import sys

SLOT_POOL_SIZE = 160  # instance_slot_count(5)
OUT_POOL_SIZE = 32
EXPIRATION = 600
COMPONENTS = 5  # the first is the merged part
PASSES = 3


class Pool:
    """A pool of variables (or resource elements) as XXMI keeps it."""

    def __init__(self, size, index_type, allocate=True, expiration=None, refresh_on_read=False, reset=True):
        self.size, self.index_type, self.allocate = size, index_type, allocate
        self.expiration, self.refresh_on_read, self.reset = expiration, refresh_on_read, reset
        self.values = [0.0] * size
        self.null = [False] * size  # a reset resource element
        self.keys = [None] * size
        self.last_update = [None] * size
        self.index_map = {}
        self.last_replacement = size - 1
        self.last_expiration_run = None

    def _postpone(self, index, assignment, frame):
        if self.expiration is not None and (assignment or self.refresh_on_read):
            self.last_update[index] = frame

    def _expire(self, frame):
        if self.expiration is None or self.last_expiration_run == frame:
            return
        self.last_expiration_run = frame
        for index in range(self.size):
            if self.last_update[index] is None or frame - self.last_update[index] <= self.expiration:
                continue
            self.last_update[index] = None
            if self.keys[index] is not None:
                del self.index_map[self.keys[index]]
                self.keys[index] = None
            if self.reset:
                self.values[index] = 0.0
                self.null[index] = True

    def index(self, key, frame, assignment=False):
        self._expire(frame)
        if self.index_type == "ring":
            index = int(key) % self.size
            self._postpone(index, assignment, frame)
            return index
        if key in self.index_map:
            index = self.index_map[key]
            self._postpone(index, assignment, frame)
            return index
        assert self.allocate or assignment
        index = (self.last_replacement + 1) % self.size
        self.last_replacement = index
        if self.keys[index] is not None:
            del self.index_map[self.keys[index]]
        self.index_map[key] = index
        self.keys[index] = key
        self._postpone(index, assignment, frame)
        return index

    def get(self, key, frame):
        return self.values[self.index(key, frame)]

    def set(self, key, value, frame):
        self.values[self.index(key, frame, assignment=True)] = value


class Failure(AssertionError):
    pass


class Runtime:
    """The ini's command lists for one character, over a model of XXMI's pools.

    ``variant``: "current", "test2" (outputs looked up by buffer identity at every use) or "test1"
    (marks kept by buffer identity, outputs that expire, slots not cleared).
    """

    def __init__(self, variant: str = "current"):
        self.variant = variant
        self.hidden = 0
        # What the mod's buffers hold, as tags: per store slot and per output buffer.
        self.header = [None] * SLOT_POOL_SIZE   # (buffer, frame) of the bones imported there
        self.record = [None] * SLOT_POOL_SIZE   # (buffer the slot had, copy, frame)
        self.owner = [None] * SLOT_POOL_SIZE    # (buffer the slot had,) when the entry was written
        self.out_held = [None] * OUT_POOL_SIZE  # (buffer, frame, valid) of the result there
        self.slot = Pool(SLOT_POOL_SIZE, "fifo", expiration=EXPIRATION, refresh_on_read=True)
        if variant == "test1":
            self.recorded = Pool(SLOT_POOL_SIZE, "fifo", expiration=EXPIRATION, refresh_on_read=True)
            self.skinned = Pool(OUT_POOL_SIZE, "fifo", expiration=EXPIRATION, refresh_on_read=True)
            self.out = Pool(OUT_POOL_SIZE, "fifo", expiration=EXPIRATION, refresh_on_read=True)
            return
        self.slot_key = Pool(SLOT_POOL_SIZE, "ring")
        self.recorded_at = Pool(SLOT_POOL_SIZE, "ring")
        self.skinned_at = Pool(SLOT_POOL_SIZE, "ring")
        self.out_at = Pool(SLOT_POOL_SIZE, "ring")
        self.out = Pool(OUT_POOL_SIZE, "fifo")  # current: the numbers (PoolZZMS_OutSlot); test2: the buffers

    def claim(self, key, frame):
        """CommandListZZMS_Claim (test1 only looks the slot up)."""
        slot = self.slot.index(key, frame)
        if self.variant != "test1" and self.slot_key.get(slot, frame) != key:
            self.slot_key.set(slot, key, frame)
            self.recorded_at.set(slot, 0, frame)
            self.skinned_at.set(slot, 0, frame)
            self.out_at.set(slot, -1, frame)
            self.header[slot] = self.record[slot] = self.owner[slot] = None
        return slot

    def check_slot(self, slot):
        """What a store slot holds must have been left by the buffer it belongs to."""
        owner = self.slot.keys[slot]
        for name, tag in (("header", self.header[slot]), ("record", self.record[slot]),
                          ("owner entry", self.owner[slot])):
            if tag is not None and tag[0] != owner:
                raise Failure("slot %d holds the %s of buffer %r, it belongs to %r" % (slot, name, tag[0], owner))

    def import_bones(self, key, frame):
        slot = self.claim(key, frame)
        self.header[slot] = (key, frame)

    def record_position(self, copy, key0, key3, frame):
        if self.variant == "test1":
            if self.recorded.get(key0, frame) == frame:
                return
            self.recorded.set(key0, frame, frame)
            slots = {self.slot.index(key0, frame), self.slot.index(key3, frame)}
        else:
            s0 = self.claim(key0, frame)
            if self.recorded_at.get(s0, frame) == frame:
                return
            self.recorded_at.set(s0, frame, frame)
            slots = {s0, self.claim(key3, frame)}
        # The record is the copy's position, written for both buffers' slots.
        for slot in slots:
            self.record[slot] = (self.slot.keys[slot], copy, frame)

    def match(self, keys, frame):
        """The matcher reads the store directly: the slots its buffers have now (no lookups)."""
        found = True
        for key in keys:
            slot = self.slot.index_map.get(key)
            if slot is None or self.header[slot] != (key, frame):
                found = False
                continue
            self.check_slot(slot)
        if not found:
            self.hidden += 1
        if found:
            for key in keys[1:]:
                self.owner[self.slot.index_map[key]] = (key,)
        return found

    def skin(self, keys, host0, host3, prev3, o0, o3, frame):
        """The skinning shader: this frame's result into o0; o3 repaired unless it is host3's valid result."""
        found = self.match(keys, frame)
        if not self.out.null[o0]:
            self.out_held[o0] = (host0, frame, found)
        if prev3 != host0:
            held = self.out_held[o3]
            if self.variant == "test1":
                repair = held is None or not held[2]
            else:
                repair = not (held is not None and held[0] == host3 and held[2])
            if repair and not self.out.null[o3]:
                self.out_held[o3] = (host0, frame, found)

    def skin_now(self, keys, host0, host3, frame):
        """CommandListZZMS_SkinNow and the draw that follows it: the outputs bound as vb0 and vb3, and the
        buffer whose result vb3 is meant to be."""
        prev3 = host0
        if self.variant == "test1":
            if self.skinned.get(host3, frame) == frame - 1:
                prev3 = host3
            if self.skinned.get(host0, frame) != frame:
                self.skinned.set(host0, frame, frame)
                self.skin(keys, host0, host3, prev3, self.out.index(host0, frame), self.out.index(prev3, frame), frame)
            return self.out.index(host0, frame), self.out.index(prev3, frame), prev3
        s0 = self.claim(host0, frame)
        if self.skinned_at.get(self.claim(host3, frame), frame) == frame - 1:
            prev3 = host3
        if self.variant == "test2":
            o0 = self.out.index(host0, frame)
            if self.skinned_at.get(s0, frame) != frame or self.out_at.get(s0, frame) != o0:
                self.skinned_at.set(s0, frame, frame)
                self.out_at.set(s0, o0, frame)
                # cs-u0 and cs-u1 look the outputs up again, and so does the draw
                self.skin(keys, host0, host3, prev3, self.out.index(host0, frame), self.out.index(prev3, frame), frame)
            return self.out.index(host0, frame), self.out.index(prev3, frame), prev3
        # Both numbers found before either is used, then skinned and drawn by number.
        o0 = self.out.index(host0, frame)
        o3 = o0
        if prev3 != host0:
            o3 = self.out.index(prev3, frame)
            if o3 == o0:
                o0 = self.out.index(host0, frame)
        # Skinned again when either output passed to another buffer since (the previous one is repaired).
        if self.skinned_at.get(s0, frame) != frame or self.out_at.get(s0, frame) != o0 * 64 + o3:
            self.skinned_at.set(s0, frame, frame)
            self.out_at.set(s0, o0 * 64 + o3, frame)
            self.skin(keys, host0, host3, prev3, o0, o3, frame)
        return o0, o3, prev3

    def check_draw(self, host0, host3, bound, frame):
        o0, o3, prev3 = bound
        if prev3 != host0 and o3 == o0:
            raise Failure("frame %d: buffer %r: its previous output was dropped for its current one" % (frame, host0))
        if self.out.null[o0]:
            raise Failure("frame %d: buffer %r drawn from a null output" % (frame, host0))
        held = self.out_held[o0]
        if held is None or held[0] != host0 or held[1] != frame:
            raise Failure("frame %d: buffer %r drawn from %r (not its result of this frame)" % (frame, host0, held))
        previous = self.out_held[o3]
        if previous is not None and previous[0] not in (host0, host3):
            raise Failure("frame %d: buffer %r got as its previous output the result of %r" % (frame, host0, previous))


def replay(variant: str = "current", frames: int = 8000, seed: int = 7, lanes: int = 2) -> int:
    """One copy always on screen and ``lanes`` others coming and going; the frames a copy was hidden."""
    rng = random.Random(seed)
    runtime = Runtime(variant)
    fresh = iter(range(1, 10 ** 9))

    def spawn():
        return {"buffers": [(next(fresh), next(fresh)) for _ in range(COMPONENTS)]}

    player = spawn()
    others = [None] * lanes
    timers = [rng.randint(5, 60) for _ in range(lanes)]
    for frame in range(1, frames + 1):
        for lane in range(lanes):
            timers[lane] -= 1
            if timers[lane] <= 0:
                others[lane] = spawn() if others[lane] is None else None
                timers[lane] = rng.randint(20, 150) if others[lane] is not None else rng.randint(5, 60)
        copies = [player] + [copy for copy in others if copy is not None]
        parity = frame % 2
        current = {id(copy): [pair[parity] for pair in copy["buffers"]] for copy in copies}
        other = {id(copy): [pair[1 - parity] for pair in copy["buffers"]] for copy in copies}
        # Skinning draws: every component of every copy, interleaved.
        imports = [(copy, k) for copy in copies for k in range(COMPONENTS)]
        rng.shuffle(imports)
        for copy, k in imports:
            runtime.import_bones(current[id(copy)][k], frame)
        # Passes: every part records where it is; the merged part is skinned once and drawn.
        for _ in range(PASSES):
            order = list(copies)
            rng.shuffle(order)
            for copy in order:
                keys, others_keys = current[id(copy)], other[id(copy)]
                for k in range(1, COMPONENTS):
                    runtime.record_position(id(copy), keys[k], others_keys[k], frame)
                runtime.record_position(id(copy), keys[0], others_keys[0], frame)
                bound = runtime.skin_now(keys, keys[0], others_keys[0], frame)
                runtime.check_draw(keys[0], others_keys[0], bound, frame)
    return runtime.hidden


def caught(variant: str, **scenario) -> str:
    try:
        replay(variant, **scenario)
    except Failure as failure:
        return str(failure)
    raise AssertionError("the %s bookkeeping was expected to fail" % variant)


def main() -> int:
    hidden = replay()
    # Copies hidden for a frame when a slot of theirs went to another buffer between their skinning
    # draws and their first draw: at most once per wrap of the slot pool.
    assert hidden < 40, hidden
    crowded = [replay(frames=3000, seed=seed, lanes=lanes) for seed in (1, 2, 3) for lanes in (3, 7)]
    old = [caught("test1"), caught("test2", frames=3000, seed=1, lanes=7)]
    print("ZZMI_MERGED_SLOT_REPLAY=PASS (hidden frames %d, crowded %s; caught: test1 %s; test2 %s)"
          % (hidden, crowded, old[0], old[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
