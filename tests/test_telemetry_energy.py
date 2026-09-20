import math
import unittest
from dataclasses import replace
from unittest.mock import Mock

from wt_overlay.contracts import FlightState, G
from wt_overlay.demo import make_demo_sample
from wt_overlay.energy import EnergyEstimator
from wt_overlay.telemetry import TelemetryClient, parse_telemetry


class TelemetryTests(unittest.TestCase):
    def sample(self, **overrides):
        state = {'valid': True, 'H, m': 1234, 'TAS, km/h': 720,
                 'IAS, km/h': 540, 'Mfuel, kg': 800, 'Ny': 4.0}
        state.update(overrides)
        return parse_telemetry(state, {'valid': True, 'type': 'su_27sm'}, 1.0)

    def test_units_and_unknown_mass(self):
        sample = self.sample(**{'thrust 1, kgf': 100, 'thrust 2, N': 1000})
        self.assertTrue(sample.valid)
        self.assertEqual(sample.tas_mps, 200)
        self.assertEqual(sample.ias_mps, 150)
        self.assertEqual(sample.altitude_m, 1234)
        self.assertEqual(sample.fuel_kg, 800)
        self.assertIsNone(sample.mass_kg)
        self.assertIsNone(sample.load_factor)
        self.assertEqual(sample.normal_load_g, 4.0)
        self.assertAlmostEqual(sample.thrust_n, 100 * G + 1000)

    def test_ambiguous_thrust_is_not_converted(self):
        self.assertIsNone(self.sample(**{'thrust 1': 100, 'thrust 2, kG': 100}).thrust_n)

    def test_invalid_core_and_nonfinite_payload(self):
        for changes in ({'valid': False}, {'TAS, km/h': -1}, {'TAS, km/h': '720'},
                        {'H, m': float('nan')}, {'AoA, deg': float('inf')}):
            self.assertFalse(self.sample(**changes).valid)
        self.assertFalse(parse_telemetry({}, {}, 0).valid)
        self.assertFalse(parse_telemetry({'valid': True, 'H, m': 1, 'TAS, km/h': 1},
                                         {'valid': True, 'army': 'tank'}, 0).valid)

    def test_local_endpoint_restrictions(self):
        for url in ('http://example.com:8111', 'https://127.0.0.1:8111',
                    'http://127.0.0.1:8111/path', 'http://u:p@127.0.0.1:8111',
                    'http://127.0.0.1:8111?x=1'):
            with self.assertRaises(ValueError):
                TelemetryClient(url)
        TelemetryClient('http://[::1]:8111')

    def test_network_failure_never_becomes_demo(self):
        client = TelemetryClient()
        client._opener = Mock()
        client._opener.open.side_effect = TimeoutError()
        sample = client.poll(4)
        self.assertFalse(sample.valid)
        self.assertEqual(sample.source, 'live')
        self.assertIn('TimeoutError', sample.notes[0])

    def test_read_size_and_json_shape_limits(self):
        client = TelemetryClient()
        response = Mock()
        client._opener = Mock()
        client._opener.open.return_value.__enter__ = Mock(return_value=response)
        client._opener.open.return_value.__exit__ = Mock(return_value=False)
        for payload in (b'x' * (256 * 1024 + 1), b'[]', b'{bad'):
            response.read.return_value = payload
            self.assertFalse(client.poll(0).valid)
        response.read.assert_called_with(256 * 1024 + 1)


class EnergyTests(unittest.TestCase):
    def state(self, t, h=1000.0, v=200.0):
        return FlightState(t, True, altitude_m=h, tas_mps=v, aircraft_id='test')

    def run_curve(self, fn):
        estimator = EnergyEstimator()
        for i in range(13):
            t = i / 10
            result = estimator.update(fn(t))
        return result

    def test_constant_speed_climb(self):
        result = self.run_curve(lambda t: self.state(t, h=1000 + 15*t))
        self.assertTrue(result.ready)
        self.assertAlmostEqual(result.sep_mps, 15)
        self.assertAlmostEqual(result.climb_mps, 15)
        self.assertAlmostEqual(result.kinetic_sep_mps, 0)
        self.assertAlmostEqual(result.energy_height_m, 1018 + 200**2/(2*G))

    def test_acceleration_uses_window_average(self):
        result = self.run_curve(lambda t: self.state(t, v=200 + 3*t))
        self.assertAlmostEqual(result.acceleration_mps2, 3)
        self.assertAlmostEqual(result.sep_mps, (200 + 3*0.6)*3/G)
        self.assertAlmostEqual(result.kinetic_sep_mps, result.sep_mps)

    def test_constant_energy_exchange(self):
        result = self.run_curve(lambda t: self.state(t, h=1000+10*t,
                                   v=math.sqrt(200**2 - 2*G*10*t)))
        self.assertAlmostEqual(result.sep_mps, 0)
        self.assertAlmostEqual(result.climb_mps, 10)
        self.assertAlmostEqual(result.kinetic_sep_mps, -10)

    def test_resets_on_discontinuity(self):
        for discontinuity in ('invalid', 'reverse', 'gap', 'aircraft', 'source'):
            estimator = EnergyEstimator()
            for t in (0, 0.1, 0.3):
                result = estimator.update(self.state(t, h=1000 + t))
            self.assertTrue(result.ready)
            next_state = self.state(0.4)
            if discontinuity == 'invalid':
                self.assertFalse(estimator.update(replace(next_state, valid=False)).ready)
                next_state = self.state(0.5)
            elif discontinuity == 'reverse':
                next_state = self.state(0.2)
            elif discontinuity == 'gap':
                next_state = self.state(10)
            elif discontinuity == 'aircraft':
                next_state = replace(next_state, aircraft_id='different')
            else:
                next_state = replace(next_state, source='demo')
            result = estimator.update(next_state)
            self.assertFalse(result.ready)
            self.assertIsNone(result.sep_mps)

    def test_nonfinite_and_warmup(self):
        estimator = EnergyEstimator()
        self.assertFalse(estimator.update(self.state(0)).ready)
        self.assertFalse(estimator.update(self.state(0.1, v=float('nan'))).ready)
        self.assertFalse(estimator.update(self.state(0.2)).ready)

    def test_overflow_does_not_escape(self):
        estimator = EnergyEstimator()
        for t, h in ((0, -1e308), (0.1, 1e308), (0.3, -1e308)):
            result = estimator.update(self.state(t, h=h))
        self.assertFalse(result.ready)
        self.assertFalse(estimator.update(self.state(0.4, v=1e308)).ready)

    def test_history_is_bounded(self):
        estimator = EnergyEstimator()
        for i in range(2000):
            estimator.update(self.state(i * 0.0001))
        self.assertLessEqual(len(estimator._samples), 512)

    def test_demo_is_explicit_and_synthetic(self):
        sample = make_demo_sample(4)
        self.assertEqual(sample.source, 'demo')
        self.assertEqual(sample.aircraft_id, 'synthetic-demo')
        self.assertIn('SYNTHETIC DEMO', sample.notes[0])
        self.assertIsNone(sample.mass_kg)
        self.assertFalse(make_demo_sample(float('nan')).valid)


if __name__ == '__main__':
    unittest.main()
