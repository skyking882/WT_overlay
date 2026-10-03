"""Reaction-time rule used by scripts/build_reaction_table.py."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_reaction_table import _reaction  # noqa: E402


def rows(kind, starts):
    return [dict(kind=kind, start_s=t, escaped=ok) for t, ok in starts]


class ReactionTests(unittest.TestCase):
    def test_union_of_kinds_continuous_from_zero(self):
        beam = rows("beam", [(0., False), (.25, True), (.5, True), (.75, False), (1., True)])
        drag = rows("drag", [(0., True), (.25, False), (.5, False), (.75, True), (1., False), (1.25, False)])
        self.assertEqual(_reaction(beam+drag, .25), (1., 1.))

    def test_gap_ends_the_window_unless_bridged(self):
        beam = rows("beam", [(0., True), (.25, True), (.5, False), (.75, True), (1., False), (1.25, False),
                             (1.5, False), (1.75, True)])
        self.assertEqual(_reaction(beam, .25), (.25, 1.))
        self.assertEqual(_reaction(beam, .25, max_gap_s=.25), (.75, .75))

    def test_immediate_reaction_hit_means_zero(self):
        beam = rows("beam", [(0., False), (.25, True), (.5, True)])
        self.assertEqual(_reaction(beam, .25), (0., 0.))


if __name__ == "__main__":
    unittest.main()
