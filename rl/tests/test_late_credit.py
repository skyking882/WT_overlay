"""Rewards that arrive after an agent is down (its missile scores later): the worker plays the match on before a
reset, and the sampler adds what arrives later to the agent's final step.

LateEnv stands in for MatchEnv's reporting: info["late_rewards"], info["tallies"], pending_credit().
"""
import unittest

import common  # noqa: F401

from rl.fake_env import FakeMatchEnv
from test_rollout import make


class LateEnv(FakeMatchEnv):
    """Every agent that dies leaves a missile that kills two steps later (+1 owed to it)."""

    def reset(self):
        self.owed = {}
        return super().reset()

    def pending_credit(self):
        return bool(self.owed)

    @property
    def over(self):
        return self.S["over"] and not self.owed

    def _pay(self):
        late = {}
        for aid in list(self.owed):
            self.owed[aid] -= 1
            if self.owed[aid] <= 0:
                del self.owed[aid]
                late[aid] = 1.0
        return late

    def step(self, actions):
        late = self._pay()
        if not actions:
            info = {"timeout": False, "events": {"kill": len(late)}}
        else:
            obs, rew, done, info = super().step(actions)
        tallies = {aid: [1, 0] for aid in late}
        if actions:
            for aid, r in rew.items():
                k, d = int(r >= 0.9 or -1.1 <= r <= -0.6), int(r <= -0.6)
                if d:
                    self.owed[aid] = 2
                if k or d:
                    t = tallies.setdefault(aid, [0, 0])
                    t[0] += k
                    t[1] += d
        if late:
            info["late_rewards"] = late
        if tallies:
            info["tallies"] = tallies
        if not actions:
            return {}, {}, {}, info
        return obs, rew, done, info


def late_cfg(n_agents):
    def fn(cfg):
        cfg.env.cls = "test_late_credit:LateEnv"
        cfg.env.streams_per_env = n_agents
        cfg.env.config.update(n_agents=n_agents, min_agents=n_agents, max_steps=40, p_background_death=0.05,
                              p_match_end=0.0)
    return fn


class SamplerCredits(unittest.TestCase):
    def test_one_agent_env_is_played_on_and_the_kill_lands_in_the_death_step(self):
        # vs-script shape: the only controlled agent dies, the worker finishes its missile before the reset.
        cfg, actor, critic, sm = make(streams=4, steps=40, seg=8, burn=4, cfg_fn=late_cfg(1))
        trades, deaths = 0, 0
        for _ in range(3):
            buf, st = sm.collect(actor)
            self.assertEqual(st.get("late_credited", 0), 0)       # nothing left over for the late path
            for o in st["outcomes"]:
                deaths += o[3]
                trades += o[1] == "trade"
                if o[3]:
                    self.assertGreaterEqual(o[2], 1)              # every death left a missile that scored
            for e in st["episodes"]:
                if e[3] == "terminal":
                    self.assertGreater(e[1], -2.0)
        self.assertGreater(deaths, 0)
        self.assertEqual(trades, deaths)

    def test_late_kills_reach_the_final_step_and_turn_losses_into_trades(self):
        cfg, actor, critic, sm = make(streams=8, steps=40, seg=8, burn=4, cfg_fn=late_cfg(2))
        credited, trades = 0, 0
        for _ in range(3):
            buf, st = sm.collect(actor)
            credited += st.get("late_credited", 0)
            trades += sum(1 for o in st["outcomes"] if o[1] == "trade")
            for e in st["episodes"]:
                self.assertEqual(len(e), 5)
        self.assertGreater(credited, 0)
        self.assertGreater(trades, 0)


if __name__ == "__main__":
    unittest.main()
