import pickle
import random
import unittest

import common
import torch

from rl import spec, wire
from rl.encode import Decoded, masks_from_flat


def nested_equal(got, want, shape_n):
    """Compare a decoded tensor (maybe wider) with the nested list `want`; padding must be False."""
    t = torch.tensor(want, dtype=torch.bool)
    sl = tuple(slice(0, d) for d in t.shape)
    if not torch.equal(got[sl], t):
        return False
    rest = got.clone()
    rest[sl] = False
    return not bool(rest.any())


class WireRoundTrip(unittest.TestCase):
    def test_masks_survive_pack_and_decode_for_every_head_and_width(self):
        rng = random.Random(0)
        obs = [common.random_obs(rng, n) for n in (0, 1, 5, 12, 64, 3)]
        dec = Decoded([wire.pack_obs(o) for o in obs])
        N = max(1, int(dec.ent_n.max()))
        masks = masks_from_flat(dec.mask, N + 1)
        for h in spec.HEAD_NAMES:
            self.assertEqual(tuple(masks[h].shape[1:]), tuple(
                ([N + 1] if spec.MASK_KIND[h] == "target" else []) +
                ([3] if spec.MASK_KIND[h] == "view" else [3, 2] if spec.MASK_KIND[h] == "view_ref" else []) +
                [N + 1 if h in spec.POINTER_HEADS else spec.CAT_SIZES[h]]), h)
            for i, o in enumerate(obs):
                self.assertTrue(nested_equal(masks[h][i], o["masks"][h], len(o["entities"])), (h, i))

    def test_float_blocks_round_trip_and_pad_with_zeros(self):
        rng = random.Random(1)
        obs = [common.random_obs(rng, n) for n in (2, 7, 0)]
        b = Decoded([wire.pack_obs(o) for o in obs]).to_batch()
        for i, o in enumerate(obs):
            n = len(o["entities"])
            self.assertEqual(int(b.ent_n[i, 0]), n)
            got = b.ent[i, 0, :n]
            self.assertLess((got - torch.tensor(o["entities"], dtype=torch.float32).reshape(n, 48)).abs().max().item() if n else 0.0, 1e-6)
            self.assertEqual(b.ent[i, 0, n:].abs().sum().item(), 0.0)
            self.assertLess((b.own[i, 0] - torch.tensor(o["own"])).abs().max().item(), 1e-6)
            self.assertEqual(int(b.truth_n[i, 0]), len(o["truth"]))

    def test_plain_types_and_pickle_protocol(self):
        o = common.random_obs(random.Random(2), 4)
        w = wire.pack_obs(o)
        self.assertLessEqual(wire.PICKLE_PROTOCOL, 5)
        for x in w:
            self.assertIsInstance(x, (bytes, int, float, str))
        self.assertEqual(pickle.loads(pickle.dumps(w, wire.PICKLE_PROTOCOL)), w)
        a = {h: i % 2 for i, h in enumerate(spec.HEAD_NAMES)}
        self.assertEqual(wire.unpack_action(wire.pack_action(a)), a)
        self.assertEqual(len(wire.pack_action(a)), 15)

    def test_contract_violations_are_reported_clearly(self):
        rng = random.Random(3)
        def with_(**kw):
            o = common.random_obs(rng, 3)
            o.update(kw)
            return o
        with self.assertRaisesRegex(wire.ContractError, "own"):
            wire.pack_obs(with_(own=[0.0] * 95))
        with self.assertRaisesRegex(wire.ContractError, "entities"):
            wire.pack_obs(with_(entities=[[0.0] * 47]))
        with self.assertRaisesRegex(wire.ContractError, "entities"):
            wire.pack_obs(with_(entities=[[0.0] * 48] * 65))
        with self.assertRaisesRegex(wire.ContractError, "truth"):
            wire.pack_obs(with_(truth=[[0.0] * 39]))
        o = with_()
        del o["masks"]["weapon"]
        with self.assertRaisesRegex(wire.ContractError, "missing head weapon"):
            wire.pack_obs(o)
        o = with_()
        o["masks"]["weapon"] = [True, True]            # 1-D instead of (n+1) x 2
        with self.assertRaisesRegex(wire.ContractError, "weapon"):
            wire.pack_obs(o)
        o = with_()
        o["masks"]["maneuver_ref"] = [[True] * 3] * 3  # K should be n+1 = 4
        with self.assertRaisesRegex(wire.ContractError, "maneuver_ref"):
            wire.pack_obs(o)
        o = with_()
        o["masks"]["weapon|target"] = [[True, True]] * 4  # the old conditional-table keys are gone
        with self.assertRaisesRegex(wire.ContractError, "unknown key"):
            wire.pack_obs(o)
        nan = with_()
        nan["own"][5] = float("nan")
        wire.pack_obs(nan)                              # not checked by default
        with self.assertRaisesRegex(wire.ContractError, "non-finite"):
            wire.pack_obs(nan, check_finite=True)

    def test_dict_and_attribute_style_observations_are_equivalent(self):
        from rl.fake_env import FakeMatchEnv
        env = FakeMatchEnv({}, 1)
        obs = env.reset()
        for o in obs.values():
            d = {k: getattr(o, k) for k in ("own", "entities", "prev_intent", "masks", "truth", "aircraft", "dt")}
            self.assertEqual(wire.pack_obs(o), wire.pack_obs(d))

    def test_non_positive_dt_falls_back_to_the_nominal_step(self):
        rng = random.Random(8)
        obs = [dict(common.random_obs(rng, 2), dt=x) for x in (0.0, -1.0, 0.5)]
        dec = Decoded([wire.pack_obs(o) for o in obs])
        self.assertEqual(dec.dt.tolist()[2], 0.5)
        self.assertAlmostEqual(dec.dt.tolist()[0], 20 / 48, 6)
        self.assertAlmostEqual(dec.dt.tolist()[1], 20 / 48, 6)

    def test_mask_layout_is_consistent(self):
        end = 0
        for h in spec.HEAD_NAMES:
            self.assertEqual(spec.MASK_OFFSET[h], end)
            end += spec.MASK_SIZE[h]
        self.assertEqual(end, spec.MASK_BYTES)
        self.assertEqual(spec.mask_shape("maneuver", 7), (3, 2, 11))
        self.assertEqual(spec.mask_shape("weapon", 7), (8, 2))
        self.assertEqual(spec.mask_shape("view_object", 7), (3, 8))
        self.assertEqual(spec.mask_shape("target", 7), (8,))
        self.assertEqual(spec.mask_shape("kb_roll", 7), (3, 3))


if __name__ == "__main__":
    unittest.main()
