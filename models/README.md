# Frozen RL policies

Weights-only exports (actor, critic, training cfg, starts, aircraft list; no optimizer or league state) of the
checkpoints trained on workstation with the code of this branch. Load the actor with `rl.eval_replay.load_actor`;
`rl.eval_replay` reads the stored `cfg.env.config` for evaluation.

| File | Source | Setting | Result (vs fixed top-tier scripts) |
|---|---|---|---|
| `s1_1v1_r304.pt` | `s1_league/ppo/ckpt_000304.pt` | 1v1, egocentric frame, 80-100 km spawn, league | 89 % win; 152-23 paired vs s1_ego2 r164 |
| `s2c_4v4_r554.pt` | `s2c_4v4/ppo/ckpt_000554.pt` | 4v4 from r304, opening-climb kickstart, anti-spam rewards off | all-4 vs top scripts (population v2): 56 % win, exchange 1.69, 100 games |

sha256
- `s1_1v1_r304.pt` c5583e034beba85f4e138a38d1f3591c3fd69e3fe75826d5580d7c98aa6c74fe
- `s2c_4v4_r554.pt` 5fe99275c51c471e1feea3720767cbd367ad0a5d16b4d6607d0543d5e9deae40

Note: since 2026-10-07, leaving free look is subject to the 5 % rejection by default; r304 was trained before that,
so re-running it on this code gives slightly different numbers from the ones above.
