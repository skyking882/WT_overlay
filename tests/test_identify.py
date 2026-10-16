"""Aircraft type from /state when 8111 /indicators is invalid (wt_overlay.identify)."""

from types import SimpleNamespace
import unittest

from wt_overlay.identify import AircraftGuess, _index, state_signature


def sample(t, fuel0, thrust_each, engines, throttle, altitude, tas_kmh):
    raw = {"Mfuel0, kg": fuel0, "throttle 1, %": throttle}
    raw.update({f"thrust {i}, kgs": thrust_each for i in range(1, engines+1)})
    return SimpleNamespace(time_s=t, raw_state=raw, altitude_m=altitude, tas_mps=tas_kmh/3.6)


class IdentifyTests(unittest.TestCase):
    def run_samples(self, guess, *args, count=6):
        result = None
        for k in range(count):
            result = guess.update(sample(.6*k, *args))
        return result

    def test_signature(self):
        self.assertEqual(state_signature({"Mfuel0, kg": 7110, "thrust 1, kgs": 10, "thrust 2, kgs": 20,
                                          "throttle 1, %": 110}), (7110., 2, 30., 110.))
        self.assertIsNone(state_signature({"thrust 1, kgs": 10}))

    def test_typhoon_from_fuel_and_thrust(self):
        # Live sample 2026-10-16: 4221 m, 1446 km/h, both engines 10170 kgf at 110 %.
        guess = AircraftGuess()
        found = self.run_samples(guess, 7110, 10170, 2, 110, 4221., 1446.)
        self.assertTrue(found.startswith("ef_2000"), found)

    def test_j16_separated_from_its_family_by_thrust(self):
        # Live sample 2026-10-16: 2888 m, 1013 km/h, both engines 6561 kgf at 100 %.
        guess = AircraftGuess()
        self.assertIsNone(guess.update(sample(0., 9400, 6561, 2, 100, 2888., 1013.)))   # needs samples
        self.assertEqual(self.run_samples(guess, 9400, 6561, 2, 100, 2888., 1013.), "j_16")

    def test_unique_fuel_is_immediate(self):
        counts = {}
        for aircraft, fuel, engines in _index():
            counts.setdefault((round(fuel), engines), []).append(aircraft)
        (fuel, engines), ids = next((k, v) for k, v in counts.items() if len(v) == 1)
        self.assertEqual(AircraftGuess().update(sample(0., fuel, 5000, engines, 100, 3000., 900.)), ids[0])

    def test_no_match_and_no_thrust_explain(self):
        guess = AircraftGuess()
        self.assertIsNone(guess.update(sample(0., 1234, 5000, 2, 100, 3000., 900.)))
        self.assertIn("1234", guess.note)
        bare = SimpleNamespace(time_s=0., raw_state={"H, m": 1}, altitude_m=1., tas_mps=1.)
        self.assertIsNone(guess.update(bare))
        self.assertIn("Mfuel0", guess.note)

    def test_changing_throttle_is_not_sampled(self):
        guess = AircraftGuess()
        for k in range(8):
            guess.update(sample(.6*k, 9400, 6561, 2, 100 if k % 2 else 90, 2888., 1013.))
        self.assertIsNone(guess.aircraft_id)


if __name__ == "__main__":
    unittest.main()
